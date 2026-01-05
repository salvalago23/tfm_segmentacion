import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union, List
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class ISIC2018Dataset(Dataset):
    """
    Dataset personalizado para ISIC 2018
    Soporta carga de imágenes y máscaras, aumentación en tiempo real
    """
    
    def __init__(self, 
                 images_dir: Union[str, Path],
                 masks_dir: Optional[Union[str, Path]] = None,
                 phase: str = 'train',
                 target_size: Tuple[int, int] = (256, 256),
                 augment: bool = True,
                 debug: bool = False):
        """
        Args:
            images_dir: Directorio con imágenes (.npy o .jpg)
            masks_dir: Directorio con máscaras (.npy o .png) - None para test
            phase: 'train', 'val', 'test'
            target_size: Tamaño objetivo (alto, ancho)
            augment: Si aplicar aumentación (solo para train)
            debug: Modo debug (muestra información)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.phase = phase
        self.target_size = target_size
        self.augment = augment and (phase == 'train')
        self.debug = debug
        
        # Obtener lista de imágenes
        self.image_ids = self._get_image_ids()
        
        if debug:
            print(f"Dataset {phase}: {len(self.image_ids)} imágenes")
            if self.masks_dir:
                print(f"Máscaras disponibles: {self.masks_dir.exists()}")
        
        # Transformaciones
        self.transform = self._get_transforms()
        
    def _get_image_ids(self) -> List[str]:
        """Obtiene IDs de imágenes disponibles"""
        # Buscar archivos .npy (procesados) o .jpg (raw)
        npy_files = list(self.images_dir.glob("*.npy"))
        jpg_files = list(self.images_dir.glob("*.jpg"))
        
        if npy_files:
            # Usar archivos .npy procesados
            return [f.stem for f in npy_files]
        elif jpg_files:
            # Usar archivos .jpg raw
            return [f.stem for f in jpg_files]
        else:
            raise FileNotFoundError(f"No se encontraron imágenes en {self.images_dir}")
    
    def _get_transforms(self) -> A.Compose:
        """Configura transformaciones según la fase"""
        
        if self.phase == 'train' and self.augment:
            # Transformaciones para entrenamiento (con aumentación)
            return A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                
                # Aumentaciones espaciales
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.Affine(translate_percent=0.1, scale=0.2, p=0.5),
                
                # Aumentaciones de color
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                          contrast_limit=0.2, 
                                          p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, 
                                     sat_shift_limit=30, 
                                     val_shift_limit=20, 
                                     p=0.5),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                
                # Normalización y conversión a tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        else:
            # Transformaciones para validación/test (solo resize + normalización)
            return A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def _load_image(self, image_id: str) -> np.ndarray:
        """Carga una imagen por ID"""
        # Intentar cargar .npy primero
        npy_path = self.images_dir / f"{image_id}.npy"
        if npy_path.exists():
            image = np.load(npy_path)
            # Si la imagen ya está normalizada [0, 1], convertir a uint8
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            return image
        
        # Si no hay .npy, cargar .jpg
        jpg_path = self.images_dir / f"{image_id}.jpg"
        if jpg_path.exists():
            image = cv2.imread(str(jpg_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        
        raise FileNotFoundError(f"No se encontró imagen para ID: {image_id}")
    
    def _load_mask(self, image_id: str) -> Optional[np.ndarray]:
        """Carga una máscara por ID (si existe)"""
        if not self.masks_dir:
            return None
        
        # Intentar cargar .npy primero
        npy_path = self.masks_dir / f"{image_id}.npy"
        if npy_path.exists():
            mask = np.load(npy_path)
            # Asegurar que sea binaria y uint8
            mask = (mask > 0.5).astype(np.uint8)
            return mask
        
        # Si no hay .npy, cargar .png
        png_path = self.masks_dir / f"{image_id}_segmentation.png"
        if png_path.exists():
            mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
            return mask
        
        if self.debug:
            print(f"No se encontró máscara para ID: {image_id}")
        return None
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Obtiene un elemento del dataset"""
        image_id = self.image_ids[idx]
        
        try:
            # Cargar imagen y máscara
            image = self._load_image(image_id)
            mask = self._load_mask(image_id)
            
            if self.debug and idx == 0:
                print(f"Imagen {image_id}: shape={image.shape}, dtype={image.dtype}")
                if mask is not None:
                    print(f"Máscara {image_id}: shape={mask.shape}, cobertura={mask.mean():.2%}")
            
            # Aplicar transformaciones
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image_tensor = transformed['image']
                
                # MANEJO CORRECTO: Albumentations devuelve Tensor con ToTensorV2
                mask_transformed = transformed['mask']
                
                # Si es Tensor (por ToTensorV2), ya está en formato correcto
                if isinstance(mask_transformed, torch.Tensor):
                    mask_tensor = mask_transformed.float().unsqueeze(0)
                else:
                    # Si es numpy array (sin ToTensorV2)
                    mask_tensor = torch.from_numpy(mask_transformed).float().unsqueeze(0)
                    
            else:
                # Para test set (sin máscaras)
                transformed = self.transform(image=image)
                image_tensor = transformed['image']
                mask_tensor = torch.zeros(1, *self.target_size, dtype=torch.float32)
            
            # Crear diccionario de salida
            sample = {
                'image': image_tensor,
                'mask': mask_tensor,
                'image_id': image_id,
                'original_size': image.shape[:2]
            }
            
            return sample
            
        except Exception as e:
            print(f"Error cargando {image_id}: {e}")
            # Devolver un sample vacío para evitar romper el entrenamiento
            return self._get_empty_sample()
    
    def _get_empty_sample(self) -> Dict[str, Any]:
        """Devuelve un sample vacío en caso de error"""
        return {
            'image': torch.zeros(3, self.target_size[0], self.target_size[1]),
            'mask': torch.zeros(1, self.target_size[0], self.target_size[1]),
            'image_id': 'error',
            'original_size': self.target_size
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calcula pesos de clase para balancear el dataset
        Útil para pérdidas ponderadas
        """
        if not self.masks_dir:
            return torch.tensor([1.0, 1.0])
        
        print("Calculando pesos de clase...")
        total_pixels = 0
        foreground_pixels = 0
        
        for image_id in self.image_ids[:100]:  # Muestrear 100 imágenes
            mask = self._load_mask(image_id)
            if mask is not None:
                total_pixels += mask.size
                foreground_pixels += mask.sum()
        
        if total_pixels == 0:
            return torch.tensor([1.0, 1.0])
        
        background_pixels = total_pixels - foreground_pixels
        
        # Peso inversamente proporcional a la frecuencia
        weight_background = total_pixels / (2 * background_pixels)
        weight_foreground = total_pixels / (2 * foreground_pixels)
        
        print(f"  Background pixels: {background_pixels}")
        print(f"  Foreground pixels: {foreground_pixels}")
        print(f"  Weight background: {weight_background:.4f}")
        print(f"  Weight foreground: {weight_foreground:.4f}")
        
        return torch.tensor([weight_background, weight_foreground])