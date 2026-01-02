"""
Script para extraer características ABCDE de todo el dataset usando modelos entrenados
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import cv2

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from clinical_analysis.abcde_analyzer import ABCDEAnalyzer
from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.residual_unet import ResidualUNet


def load_model(model_path: str, model_type: str, device: str = 'cuda'):
    """Carga modelo entrenado"""
    
    # Seleccionar arquitectura
    if model_type == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif model_type == 'attention_unet':
        model = AttentionUNet(in_channels=3, out_channels=1)
    elif model_type == 'residual_unet':
        model = ResidualUNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Cargar pesos (weights_only=False para compatibilidad con checkpoints antiguos)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Manejar diferentes formatos de checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path: str, target_size: tuple = (256, 256)):
    """Preprocesa imagen para el modelo"""
    # Cargar imagen
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    
    # Convertir a numpy y normalizar
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Para PyTorch: (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    
    return image_np, image_tensor


def predict_mask(model, image_tensor, device='cuda'):
    """Genera predicción de segmentación"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        
        # Aplicar sigmoid y umbral
        mask = torch.sigmoid(output)
        mask = (mask > 0.5).float()
        
        # Convertir a numpy
        mask_np = mask.squeeze().cpu().numpy()
    
    return mask_np


def extract_features_from_dataset(
    data_dir: str,
    model_path: str,
    model_type: str,
    output_csv: str,
    split: str = 'test',
    pixels_per_mm: float = 10.0,
    device: str = 'cuda'
):
    """
    Extrae características ABCDE de todas las imágenes del dataset
    
    Args:
        data_dir: Directorio raíz de datos
        model_path: Ruta al modelo entrenado
        model_type: Tipo de modelo ('unet', 'attention_unet', 'residual_unet')
        output_csv: Ruta donde guardar el CSV con características
        split: Split del dataset ('train', 'val', 'test')
        pixels_per_mm: Factor de conversión
        device: 'cuda' o 'cpu'
    """
    
    print(f"Loading model: {model_type}")
    model = load_model(model_path, model_type, device)
    
    print("Initializing ABCDE Analyzer")
    analyzer = ABCDEAnalyzer(pixels_per_mm=pixels_per_mm)
    
    # Rutas de datos
    images_dir = Path(data_dir) / 'raw' / f'isic2018_{split}' / 'images'
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Obtener lista de imágenes
    image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    
    print(f"Found {len(image_files)} images in {split} split")
    
    # Lista para almacenar resultados
    results = []
    
    # Procesar cada imagen
    for img_path in tqdm(image_files, desc=f"Extracting features ({split})"):
        try:
            # Cargar y preprocesar
            image_np, image_tensor = preprocess_image(str(img_path))
            
            # Predecir máscara
            mask = predict_mask(model, image_tensor, device)
            
            # Convertir imagen de vuelta a RGB uint8 para análisis
            image_rgb = (image_np * 255).astype(np.uint8)
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Extraer características ABCDE
            features = analyzer.analyze(image_rgb, mask_uint8)
            
            # Añadir metadatos
            result = {
                'image_id': img_path.stem,
                'split': split,
                'model_type': model_type,
                **features.to_dict()
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue
    
    # Convertir a DataFrame y guardar
    df = pd.DataFrame(results)
    
    # Crear directorio de salida si no existe
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    
    print(f"\nFeatures extracted successfully!")
    print(f"Total images processed: {len(df)}")
    print(f"Results saved to: {output_csv}")
    
    # Mostrar estadísticas básicas
    print("\n=== Basic Statistics ===")
    print(f"Mean Asymmetry: {df['asymmetry_score'].mean():.3f}")
    print(f"Mean Border Irregularity: {df['border_irregularity'].mean():.3f}")
    print(f"Mean Number of Colors: {df['num_colors'].mean():.1f}")
    print(f"Mean Diameter (mm): {df['diameter_mm'].mean():.2f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Extract ABCDE features from dataset')
    
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root data directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['unet', 'attention_unet', 'residual_unet'],
                       help='Type of model architecture')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--pixels_per_mm', type=float, default=10.0,
                       help='Pixels per millimeter conversion factor')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Ejecutar extracción
    df = extract_features_from_dataset(
        data_dir=args.data_dir,
        model_path=args.model_path,
        model_type=args.model_type,
        output_csv=args.output_csv,
        split=args.split,
        pixels_per_mm=args.pixels_per_mm,
        device=args.device
    )


if __name__ == '__main__':
    main()