"""
Aplicación web de segmentación de lesiones cutáneas - Backend
Servidor FastAPI para inferencia de modelos y gestión de pacientes
"""

import os
import sys
import time
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
import io
import base64

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

 # Añadir la raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import UNet, AttentionUNet, ResidualUNet, UNetPlusPlus, DeepLabV3Plus, TransUNet
from clinical_analysis import ABCDAnalyzer



# ============================================================================
# Configuración
# ============================================================================

class Config:
    MODEL_DIR = PROJECT_ROOT / "notebooks" / "experiments" / "notebook_experiments"
    DATA_DIR = Path(__file__).parent / "data"
    PATIENTS_DB = DATA_DIR / "patients.json"
    RESULTS_DIR = DATA_DIR / "results"
    
    # Configuraciones de modelos
    AVAILABLE_MODELS = {
        "unet": {
            "name": "U-Net Básica",
            "class": UNet,
            "path": MODEL_DIR / "u-net_básica" / "best_model.pth",
            "kwargs": {"in_channels": 3, "out_channels": 1, "features": [64, 128, 256, 512]}
        },
        "attention_unet": {
            "name": "Attention U-Net",
            "class": AttentionUNet,
            "path": MODEL_DIR / "attention_u-net" / "best_model.pth",
            "kwargs": {"in_channels": 3, "out_channels": 1, "features": [64, 128, 256, 512]}
        },
        "residual_unet": {
            "name": "Residual U-Net",
            "class": ResidualUNet,
            "path": MODEL_DIR / "residual_u-net" / "best_model.pth",
            "kwargs": {"in_channels": 3, "out_channels": 1, "features": [64, 128, 256, 512]}
        },
        "unetpp": {
            "name": "U-Net++",
            "class": UNetPlusPlus,
            "path": MODEL_DIR / "u-net++" / "best_model.pth",
            "kwargs": None
        },
        "deeplabv3": {
            "name": "DeepLabV3+",
            "class": DeepLabV3Plus,
            "path": MODEL_DIR / "deeplabv3+" / "best_model.pth",
            "kwargs": None
        },
        "transunet": {
            "name": "TransUNet",
            "class": TransUNet,
            "path": MODEL_DIR / "transunet" / "best_model.pth",
            "kwargs": None
        }
    }
    
    # Preprocesado de imágenes
    TARGET_SIZE = (256, 256)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    THRESHOLD = 0.5



# ============================================================================
# Modelos de datos
# ============================================================================

class Patient(BaseModel):
    id: str
    name: str
    created_at: str
    notes: Optional[str] = ""

class ABCDFeatures(BaseModel):
    """Características dermatoscópicas ABCD"""
    # A - Asimetría
    asymmetry_score: float
    asymmetry_x: float
    asymmetry_y: float
    asymmetry_risk: str  # Low/Medium/High
    
    # B - Borde
    border_irregularity: float
    compactness: float
    border_risk: str  # Low/Medium/High
    
    # C - Color
    num_colors: int
    color_variance: float
    has_blue_gray: bool
    has_white: bool
    has_red: bool
    color_risk: str  # Low/Medium/High
    
    # D - Diámetro
    diameter_mm: float
    area_mm2: float
    diameter_risk: str  # Low/Medium/High
    
    # Evaluación global de riesgo
    overall_risk: str  # Low/Medium/High
    risk_score: float  # 0-10

class PredictionResult(BaseModel):
    id: str
    patient_id: str
    model_used: str
    inference_time_ms: float
    lesion_coverage: float
    created_at: str
    original_image: str  # Base64
    segmentation_mask: str  # Base64
    overlay_image: str  # Base64
    abcd_features: Optional[ABCDFeatures] = None  # Análisis clínico

class PredictionResponse(BaseModel):
    success: bool
    result: Optional[PredictionResult] = None
    error: Optional[str] = None



# ============================================================================
# Gestión de modelos
# ============================================================================

class ModelManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_models = {}
        print(f"🔧 Device: {self.device}")
    
    def load_model(self, model_key: str):
        """Carga un modelo por clave"""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        if model_key not in Config.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        
        config = Config.AVAILABLE_MODELS[model_key]
        
        if not config["path"].exists():
            raise FileNotFoundError(f"Model file not found: {config['path']}")
        
        print(f"Loading {config['name']}...")
        
        # Cargar el checkpoint primero para obtener la configuración
        checkpoint = torch.load(config["path"], map_location=self.device, weights_only=False)
        
        # Determinar los kwargs del modelo - preferir la configuración del checkpoint, si no usar por defecto
        model_kwargs = config["kwargs"]
        
        if model_kwargs is None:
            # Se deben obtener los kwargs del checkpoint
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                model_kwargs = saved_config.get('model_kwargs', {})
                print(f"   Using kwargs from checkpoint: {model_kwargs}")
            else:
                # Usar valores por defecto según la clase del modelo
                model_class = config["class"]
                if model_class == UNetPlusPlus:
                    model_kwargs = {"in_channels": 3, "out_channels": 1, 
                                   "features": [64, 128, 256, 512], "deep_supervision": False}
                elif model_class == DeepLabV3Plus:
                    model_kwargs = {"in_channels": 3, "out_channels": 1,
                                   "features": [64, 128, 256, 512], "aspp_channels": 256,
                                   "atrous_rates": [6, 12, 18]}
                elif model_class == TransUNet:
                    model_kwargs = {"in_channels": 3, "out_channels": 1,
                                   "img_size": 256, "base_features": 32,
                                   "embed_dim": 256, "num_heads": 8, "num_layers": 4}
                else:
                    model_kwargs = {"in_channels": 3, "out_channels": 1, 
                                   "features": [64, 128, 256, 512]}
                print(f"   Using default kwargs: {model_kwargs}")
        
        # Crear instancia del modelo
        try:
            model = config["class"](**model_kwargs)
        except Exception as e:
            print(f"Error creating model with kwargs {model_kwargs}: {e}")
            raise
        
        # Cargar pesos
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise
        
        model.to(self.device)
        model.eval()
        
        self.loaded_models[model_key] = model
        print(f"{config['name']} loaded successfully")
        
        return model
    
    def get_available_models(self) -> List[dict]:
        """Get list of available models"""
        available = []
        for key, config in Config.AVAILABLE_MODELS.items():
            available.append({
                "key": key,
                "name": config["name"],
                "available": config["path"].exists()
            })
        return available
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        # Redimensionar
        image = image.resize(Config.TARGET_SIZE, Image.BILINEAR)
        
        # Convertir a numpy
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Manejar escala de grises
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Normalizar
        img_array = (img_array - Config.MEAN) / Config.STD
        
        # Convertir a tensor (C, H, W)
        tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
        
        return tensor.unsqueeze(0)  # Añadir dimensión de batch
    
    def predict(self, model_key: str, image: Image.Image) -> tuple:
        """Realiza la predicción y devuelve la máscara con el tiempo de inferencia"""
        model = self.load_model(model_key)
        
        # Preprocesar
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Inferencia con temporización
        start_time = time.time()
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output)
            mask = (prob > Config.THRESHOLD).float()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Convertir a numpy
        mask_np = mask[0, 0].cpu().numpy()
        prob_np = prob[0, 0].cpu().numpy()
        
        return mask_np, prob_np, inference_time



# ============================================================================
# Gestión de base de datos (basada en JSON simple)
# ============================================================================

class DatabaseManager:
    def __init__(self):
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        if not Config.PATIENTS_DB.exists():
            self._save_db({"patients": {}, "results": {}})
    
    def _load_db(self) -> dict:
        with open(Config.PATIENTS_DB, 'r') as f:
            return json.load(f)
    
    def _save_db(self, data: dict):
        with open(Config.PATIENTS_DB, 'w') as f:
            json.dump(data, f, indent=2)
    
    # Métodos de pacientes
    def create_patient(self, name: str, notes: str = "") -> Patient:
        db = self._load_db()
        patient_id = str(uuid.uuid4())[:8]
        
        patient = Patient(
            id=patient_id,
            name=name,
            created_at=datetime.now().isoformat(),
            notes=notes
        )
        
        db["patients"][patient_id] = patient.model_dump()
        self._save_db(db)
        
        return patient
    
    def get_patient(self, patient_id: str) -> Optional[Patient]:
        db = self._load_db()
        if patient_id in db["patients"]:
            return Patient(**db["patients"][patient_id])
        return None
    
    def get_all_patients(self) -> List[Patient]:
        db = self._load_db()
        return [Patient(**p) for p in db["patients"].values()]
    
    def delete_patient(self, patient_id: str) -> bool:
        db = self._load_db()
        if patient_id in db["patients"]:
            del db["patients"][patient_id]
            # También elimina los resultados asociados
            db["results"] = {k: v for k, v in db["results"].items() 
                           if v.get("patient_id") != patient_id}
            self._save_db(db)
            return True
        return False
    
    # Métodos de resultados
    def save_result(self, result: PredictionResult):
        db = self._load_db()
        db["results"][result.id] = result.model_dump()
        self._save_db(db)
    
    def get_patient_results(self, patient_id: str) -> List[PredictionResult]:
        db = self._load_db()
        results = [PredictionResult(**r) for r in db["results"].values() 
                   if r.get("patient_id") == patient_id]
        return sorted(results, key=lambda x: x.created_at, reverse=True)
    
    def get_result(self, result_id: str) -> Optional[PredictionResult]:
        db = self._load_db()
        if result_id in db["results"]:
            return PredictionResult(**db["results"][result_id])
        return None
    
    def delete_result(self, result_id: str) -> bool:
        """Borra un resultado específico por ID"""
        db = self._load_db()
        if result_id in db["results"]:
            del db["results"][result_id]
            self._save_db(db)
            return True
        return False



# ============================================================================
# Utilidades de imagen
# ============================================================================

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convierte una imagen PIL a cadena base64"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()

def create_overlay(original: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Crea una superposición de la máscara sobre la imagen original"""
    original = original.resize(Config.TARGET_SIZE, Image.BILINEAR)
    original_np = np.array(original)
    
    if len(original_np.shape) == 2:
        original_np = np.stack([original_np] * 3, axis=-1)
    elif original_np.shape[-1] == 4:
        original_np = original_np[:, :, :3]
    
    # Crear máscara coloreada (verde para segmentación)
    overlay = original_np.copy()
    mask_bool = mask > 0.5
    
    # Aplicar tinte verde donde hay máscara
    overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * (1 - alpha), 0, 255)
    overlay[mask_bool, 1] = np.clip(overlay[mask_bool, 1] * (1 - alpha) + 255 * alpha, 0, 255)
    overlay[mask_bool, 2] = np.clip(overlay[mask_bool, 2] * (1 - alpha), 0, 255)
    
    # Añadir contorno
    from scipy import ndimage
    contour = ndimage.binary_dilation(mask_bool) ^ mask_bool
    overlay[contour] = [255, 255, 0]  # Contorno amarillo
    
    return Image.fromarray(overlay.astype(np.uint8))


def analyze_abcd_features(image: np.ndarray, mask: np.ndarray) -> ABCDFeatures:
    """
    Analiza las características dermatoscópicas ABCD de la lesión segmentada
    
    Args:
        image: Imagen RGB como array numpy (H, W, 3)
        mask: Máscara de segmentación binaria (H, W)
    
    Returns:
        ABCDFeatures con el análisis clínico
    """
    # Inicializar el analizador
    analyzer = ABCDAnalyzer(pixels_per_mm=10.0)  # Approximate conversion
    
    # Ejecutar análisis
    features = analyzer.analyze(image, mask)
    
    # Calcular niveles de riesgo según umbrales clínicos
    # A - Asimetría: Score > 0.3 es preocupante
    if features.asymmetry_score < 0.2:
        asymmetry_risk = "Bajo"
    elif features.asymmetry_score < 0.4:
        asymmetry_risk = "Medio"
    else:
        asymmetry_risk = "Alto"
    
    # B - Borde: Irregularidad > 0.5 es preocupante
    if features.border_irregularity < 0.3:
        border_risk = "Bajo"
    elif features.border_irregularity < 0.6:
        border_risk = "Medio"
    else:
        border_risk = "Alto"
    
    # C - Color: >3 colores o presencia de colores preocupantes
    color_risk_score = 0
    if features.num_colors > 3:
        color_risk_score += 1
    if features.has_blue_gray:
        color_risk_score += 2
    if features.has_white:
        color_risk_score += 1
    if features.has_red:
        color_risk_score += 1
    
    if color_risk_score <= 1:
        color_risk = "Bajo"
    elif color_risk_score <= 2:
        color_risk = "Medio"
    else:
        color_risk = "Alto"
    
    # D - Diámetro: >6mm es preocupante
    if features.diameter_mm < 5:
        diameter_risk = "Bajo"
    elif features.diameter_mm < 8:
        diameter_risk = "Medio"
    else:
        diameter_risk = "Alto"
    
    # Puntuación de riesgo global (combinación ponderada)
    risk_weights = {"Bajo": 0, "Medio": 1, "Alto": 2}
    risk_score_raw = (
        risk_weights[asymmetry_risk] * 2.0 +
        risk_weights[border_risk] * 1.5 +
        risk_weights[color_risk] * 2.0 +
        risk_weights[diameter_risk] * 1.0
    )
    
    # Normalizar a escala 0-10
    risk_score = min(10, (risk_score_raw / 13.0) * 10)
    
    # Nivel de riesgo global
    if risk_score < 3:
        overall_risk = "Bajo"
    elif risk_score < 6:
        overall_risk = "Medio"
    else:
        overall_risk = "Alto"
    
    return ABCDFeatures(
        asymmetry_score=round(features.asymmetry_score, 3),
        asymmetry_x=round(features.asymmetry_x, 3),
        asymmetry_y=round(features.asymmetry_y, 3),
        asymmetry_risk=asymmetry_risk,
        border_irregularity=round(features.border_irregularity, 3),
        compactness=round(features.compactness, 3),
        border_risk=border_risk,
        num_colors=features.num_colors,
        color_variance=round(features.color_variance, 2),
        has_blue_gray=features.has_blue_gray,
        has_white=features.has_white,
        has_red=features.has_red,
        color_risk=color_risk,
        diameter_mm=round(features.diameter_mm, 2),
        area_mm2=round(features.area_mm2, 2),
        diameter_risk=diameter_risk,
        overall_risk=overall_risk,
        risk_score=round(risk_score, 1)
    )



# ============================================================================
# Aplicación FastAPI
# ============================================================================

 # Instancias globales
model_manager: ModelManager = None
db_manager: DatabaseManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor del ciclo de vida de la aplicación"""
    global model_manager, db_manager
    
    print("Starting Skin Lesion Segmentation App...")
    model_manager = ModelManager()
    db_manager = DatabaseManager()
    
    yield
    
    print("Shutting down...")

app = FastAPI(
    title="Skin Lesion Segmentation",
    description="Web application for skin lesion segmentation using deep learning models",
    version="1.0.0",
    lifespan=lifespan
)

 # Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 # Servir frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")



# ============================================================================
# Endpoints de la API
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Servir frontend"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return "<h1>Skin Lesion Segmentation API</h1><p>Frontend not found. Visit /docs for API documentation.</p>"

@app.get("/api/health")
async def health_check():
    """Endpoint de comprobación de salud"""
    return {"status": "healthy", "device": str(model_manager.device)}

@app.get("/api/models")
async def get_models():
    """Obtener lista de modelos disponibles"""
    return {"models": model_manager.get_available_models()}

 # Endpoints de pacientes
@app.post("/api/patients")
async def create_patient(name: str = Form(...), notes: str = Form("")):
    """Crear un nuevo paciente"""
    patient = db_manager.create_patient(name, notes)
    return {"success": True, "patient": patient.model_dump()}

@app.get("/api/patients")
async def get_patients():
    """Obtener todos los pacientes"""
    patients = db_manager.get_all_patients()
    return {"patients": [p.model_dump() for p in patients]}

@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str):
    """Obtener un paciente específico"""
    patient = db_manager.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"patient": patient.model_dump()}

@app.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: str):
    """Delete a patient"""
    if db_manager.delete_patient(patient_id):
        return {"success": True}
    raise HTTPException(status_code=404, detail="Patient not found")

@app.get("/api/patients/{patient_id}/results")
async def get_patient_results(patient_id: str):
    """Get all results for a patient"""
    results = db_manager.get_patient_results(patient_id)
    return {"results": [r.model_dump() for r in results]}

@app.delete("/api/results/{result_id}")
async def delete_result(result_id: str):
    """Delete a specific analysis result"""
    if db_manager.delete_result(result_id):
        return {"success": True}
    raise HTTPException(status_code=404, detail="Result not found")

# Prediction endpoint
@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("unet"),
    patient_id: str = Form(...)
):
    """Run segmentation prediction on uploaded image"""
    try:
        # Validate patient
        patient = db_manager.get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Run prediction
        mask, prob, inference_time = model_manager.predict(model, image)
        
        # Calculate lesion coverage
        lesion_coverage = float(mask.mean()) * 100
        
        # Create images for response
        original_resized = image.resize(Config.TARGET_SIZE, Image.BILINEAR)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        overlay_image = create_overlay(image, mask)
        
        # Perform ABCD clinical analysis
        abcd_features = None
        try:
            abcd_features = analyze_abcd_features(np.array(original_resized), mask)
        except Exception as e:
            print(f"ABCD analysis error: {e}")
        
        # Create result
        result = PredictionResult(
            id=str(uuid.uuid4())[:8],
            patient_id=patient_id,
            model_used=Config.AVAILABLE_MODELS[model]["name"],
            inference_time_ms=round(inference_time, 2),
            lesion_coverage=round(lesion_coverage, 2),
            created_at=datetime.now().isoformat(),
            original_image=image_to_base64(original_resized),
            segmentation_mask=image_to_base64(mask_image),
            overlay_image=image_to_base64(overlay_image),
            abcd_features=abcd_features
        )
        
        # Save result
        db_manager.save_result(result)
        
        return PredictionResponse(success=True, result=result)
    
    except HTTPException:
        raise
    except Exception as e:
        return PredictionResponse(success=False, error=str(e))


# ============================================================================
# Run server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
