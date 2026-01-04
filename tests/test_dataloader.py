#!/usr/bin/env python3
"""
Script para probar el DataLoader
"""

import sys
from pathlib import Path

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.data_preparation.data_loader import MedicalDataLoader

def test_dataloader():
    """Prueba completa del DataLoader"""
    
    print("🧪 PROBANDO DATALOADER")
    print("="*50)
    
    # Configurar DataLoader
    data_loader = MedicalDataLoader(
        base_path="data",
        batch_size=4,
        target_size=(256, 256),
        num_workers=2,  # Para prueba, usar pocos workers
        pin_memory=torch.cuda.is_available(),
        use_class_balancing=True,
        debug=True
    )
    
    # Crear datasets
    print("\n1. Creando datasets...")
    try:
        data_loader.create_datasets(use_processed=True, augment_train=True)
    except Exception as e:
        print(f"⚠️  Usando datos raw: {e}")
        data_loader.create_datasets(use_processed=False, augment_train=True)
    
    # Crear DataLoaders
    print("\n2. Creando DataLoaders...")
    data_loader.create_dataloaders(shuffle_train=True)
    
    # Obtener pesos de clase
    print("\n3. Obteniendo pesos de clase...")
    class_weights = data_loader.get_class_weights()
    if class_weights is not None:
        print(f"   Pesos de clase: {class_weights}")
    
    # Visualizar batch de entrenamiento
    print("\n4. Visualizando batch de entrenamiento...")
    data_loader.visualize_batch(num_samples=4, phase='train')
    
    # Visualizar batch de validación
    print("\n5. Visualizando batch de validación...")
    data_loader.visualize_batch(num_samples=4, phase='val')
    
    # Probar iteración completa
    print("\n6. Probando iteración sobre train loader...")
    try:
        for i, batch in enumerate(data_loader.train_loader):
            images = batch['image']
            masks = batch['mask']
            
            print(f"   Batch {i+1}:")
            print(f"     Images: {images.shape}, dtype: {images.dtype}")
            print(f"     Masks: {masks.shape}, dtype: {masks.dtype}")
            print(f"     Image IDs: {batch['image_id'][:3]}...")
            
            if i >= 2:  # Solo probar 3 batches
                break
    except Exception as e:
        print(f"❌ Error durante iteración: {e}")
    
    # Estadísticas de cobertura
    print("\n7. Calculando estadísticas de cobertura...")
    if data_loader.train_dataset:
        total_coverage = 0
        count = 0
        
        for i in range(min(100, len(data_loader.train_dataset))):
            sample = data_loader.train_dataset[i]
            mask = sample['mask']
            coverage = mask.mean().item()
            total_coverage += coverage
            count += 1
        
        if count > 0:
            print(f"   Cobertura media en {count} muestras: {total_coverage/count:.3%}")
    
    print("\n✅ Prueba completada exitosamente!")

if __name__ == "__main__":
    test_dataloader()