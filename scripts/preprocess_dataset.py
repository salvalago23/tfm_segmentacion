#!/usr/bin/env python3
"""
Script para preprocesar todo el dataset ISIC 2018
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preparation.preprocessing import MedicalImagePreprocessor

def preprocess_dataset(input_base: Path, output_base: Path, target_size: tuple = (256, 256)):
    """
    Preprocesa todo el dataset y guarda en formato .npy
    
    Args:
        input_base: Directorio base con datos raw
        output_base: Directorio base para datos procesados
        target_size: Tamaño objetivo (alto, ancho)
    """
    
    preprocessor = MedicalImagePreprocessor(target_size=target_size)
    
    datasets = ['train', 'val', 'test']
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Preprocesando dataset: {dataset}")
        print(f"{'='*50}")
        
        # Crear directorios de salida
        dataset_output = output_base / dataset
        (dataset_output / "images").mkdir(parents=True, exist_ok=True)
        
        if dataset != 'test':
            (dataset_output / "masks").mkdir(parents=True, exist_ok=True)
        
        # Rutas de entrada
        input_images = input_base / f"isic2018_{dataset}" / "images"
        input_masks = None if dataset == 'test' else input_base / f"isic2018_{dataset}" / "masks"
        
        # Obtener lista de imágenes
        image_files = list(input_images.glob("*.jpg"))
        
        if not image_files:
            print(f"⚠️  No hay imágenes en {input_images}")
            continue
        
        # Procesar cada imagen
        for img_path in tqdm(image_files, desc=f"Procesando {dataset}"):
            img_id = img_path.stem
            
            # Determinar ruta de máscara
            mask_path = None
            if input_masks:
                mask_path = input_masks / f"{img_id}_segmentation.png"
                if not mask_path.exists():
                    print(f"⚠️  Máscara no encontrada para {img_id}, saltando...")
                    continue
            
            try:
                # Procesar par imagen-máscara
                image_proc, mask_proc = preprocessor.process_single_pair(
                    image_path=img_path,
                    mask_path=mask_path,
                    normalize=True,
                    augment=False  # La aumentación se hará durante el entrenamiento
                )
                
                # Guardar imagen procesada
                img_output = dataset_output / "images" / f"{img_id}.npy"
                np.save(img_output, image_proc)
                
                # Guardar máscara procesada si existe
                if mask_proc is not None:
                    mask_output = dataset_output / "masks" / f"{img_id}.npy"
                    np.save(mask_output, mask_proc)
                    
            except Exception as e:
                print(f"❌ Error procesando {img_id}: {e}")
                continue
    
    print("\n✅ Preprocesamiento completado!")

def main():
    # Configuración
    BASE_DIR = Path("data")
    INPUT_DIR = BASE_DIR / "raw"
    OUTPUT_DIR = BASE_DIR / "processed"
    TARGET_SIZE = (256, 256)  # Basado en recomendaciones del EDA
    
    # Ejecutar preprocesamiento
    preprocess_dataset(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)
    
    # Mostrar estadísticas finales
    print("\n" + "="*50)
    print("ESTADÍSTICAS FINALES DEL PREPROCESAMIENTO")
    print("="*50)
    
    for dataset in ['train', 'val', 'test']:
        dataset_path = OUTPUT_DIR / dataset / "images"
        n_images = len(list(dataset_path.glob("*.npy"))) if dataset_path.exists() else 0
        
        if dataset != 'test':
            masks_path = OUTPUT_DIR / dataset / "masks"
            n_masks = len(list(masks_path.glob("*.npy"))) if masks_path.exists() else 0
            print(f"{dataset.upper()}: {n_images} imágenes, {n_masks} máscaras")
        else:
            print(f"{dataset.upper()}: {n_images} imágenes (sin máscaras)")

if __name__ == "__main__":
    main()