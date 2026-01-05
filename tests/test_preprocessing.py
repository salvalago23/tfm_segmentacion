"""
Verifica que el preprocesamiento se realizó correctamente
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def test_preprocessing(processed_dir: Path, n_samples: int = 5):
    """Verifica visualmente el preprocesamiento"""
    
    for dataset in ['train', 'val']:
        print(f"\n🔍 Verificando dataset: {dataset}")
        
        images_dir = processed_dir / dataset / "images"
        masks_dir = processed_dir / dataset / "masks"
        
        if not images_dir.exists() or not masks_dir.exists():
            print(f"⚠️  Dataset {dataset} no encontrado")
            continue
        
        # Obtener algunos archivos
        image_files = list(images_dir.glob("*.npy"))[:n_samples]
        
        # Crear figura
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        fig.suptitle(f"Verificación Preprocesamiento - {dataset.upper()}", fontsize=16, y=1.02)
        
        for i, img_file in enumerate(image_files):
            img_id = img_file.stem
            
            # Cargar datos procesados
            image = np.load(img_file)
            mask_file = masks_dir / f"{img_id}.npy"
            mask = np.load(mask_file) if mask_file.exists() else None
            
            # Mostrar imagen original procesada
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f"Imagen: {img_id[:10]}...")
            axes[i, 0].axis('off')
            
            # Mostrar máscara
            if mask is not None:
                axes[i, 1].imshow(mask, cmap='gray')
                axes[i, 1].set_title(f"Máscara (cobertura: {mask.mean():.1%})")
                axes[i, 1].axis('off')
                
                # Mostrar superposición
                overlay = image.copy()
                overlay[:, :, 1] = np.where(mask > 0, overlay[:, :, 1] + 0.3, overlay[:, :, 1])
                overlay = np.clip(overlay, 0, 1)
                axes[i, 2].imshow(overlay)
                axes[i, 2].set_title("Superposición")
                axes[i, 2].axis('off')
            
            # Estadísticas
            print(f"  {img_id}: Imagen {image.shape}, min={image.min():.3f}, max={image.max():.3f}, mean={image.mean():.3f}")
            if mask is not None:
                print(f"        Máscara cobertura: {mask.mean():.2%}")
        
        plt.tight_layout()
        plt.show()

def check_statistics(processed_dir: Path):
    """Calcula estadísticas de los datos procesados"""
    
    print("\n📊 ESTADÍSTICAS DE DATOS PROCESADOS")
    print("="*40)
    
    for dataset in ['train', 'val']:
        images_dir = processed_dir / dataset / "images"
        
        if not images_dir.exists():
            continue
        
        # Calcular estadísticas sobre un subset
        image_files = list(images_dir.glob("*.npy"))[:100]
        
        if not image_files:
            continue
        
        pixel_values = []
        for img_file in image_files:
            img = np.load(img_file)
            pixel_values.extend(img.flatten())
        
        pixel_values = np.array(pixel_values)
        
        print(f"\n{dataset.upper()}:")
        print(f"  Muestras analizadas: {len(image_files)}")
        print(f"  Media píxeles: {pixel_values.mean():.4f}")
        print(f"  Std píxeles: {pixel_values.std():.4f}")
        print(f"  Min píxeles: {pixel_values.min():.4f}")
        print(f"  Max píxeles: {pixel_values.max():.4f}")

if __name__ == "__main__":
    PROCESSED_DIR = Path("data/processed")
    
    # Verificar visualmente
    test_preprocessing(PROCESSED_DIR, n_samples=3)
    
    # Calcular estadísticas
    check_statistics(PROCESSED_DIR)