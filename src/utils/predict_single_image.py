#!/usr/bin/env python3
"""
Script para realizar inferencia con un modelo entrenado en una sola imagen
Muy útil para demostraciones y pruebas rápidas
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import UNet, AttentionUNet, ResidualUNet

def load_model(model_path, model_type='auto', device='cpu'):
    """
    Carga un modelo entrenado
    """
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except:
        checkpoint = torch.load(model_path, map_location=device)
    
    # Determinar tipo de modelo
    if model_type == 'auto':
        # Inferir del nombre del archivo o checkpoint
        model_path_str = str(model_path).lower()
        if 'attention' in model_path_str:
            model_type = 'attention'
        elif 'residual' in model_path_str:
            model_type = 'residual'
        else:
            model_type = 'unet'
    
    # Crear modelo
    if model_type == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif model_type == 'attention':
        model = AttentionUNet(in_channels=3, out_channels=1)
    elif model_type == 'residual':
        model = ResidualUNet(in_channels=3, out_channels=1)
    
    # Cargar pesos
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✅ Modelo cargado: {model.__class__.__name__}")
    print(f"   Tipo: {model_type}")
    print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocesa una imagen para el modelo
    """
    # Cargar imagen
    if isinstance(image_path, str):
        image_path = Path(image_path)
    
    # Verificar que existe
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    
    # Cargar con OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        # Intentar con PIL si OpenCV falla
        img = np.array(Image.open(image_path))
        if len(img.shape) == 2:  # Si es escala de grises
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        # Convertir BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Guardar original para visualización
    original = img.copy()
    
    # Redimensionar
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalizar (mismo que durante entrenamiento)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # Convertir a tensor: [C, H, W]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    
    return img_tensor, original

def predict_mask(model, image_tensor, device='cpu', threshold=0.5):
    """
    Realiza predicción con el modelo
    """
    # Añadir dimensión de batch: [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.sigmoid(output)
        binary_mask = (probabilities > threshold).float()
    
    # Quitar dimensión de batch y convertir a numpy
    prob_np = probabilities[0, 0].cpu().numpy()  # [H, W]
    mask_np = binary_mask[0, 0].cpu().numpy()    # [H, W]
    
    return prob_np, mask_np

def visualize_results(original_image, probability_map, binary_mask, 
                     image_name, threshold=0.5, save_path=None):
    """
    Visualiza resultados de la segmentación
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Redimensionar las máscaras para que coincidan con el tamaño original
    target_size = original_image.shape[:2]  # (alto, ancho)
    
    # 1. Redimensionar probability_map y binary_mask al tamaño original
    probability_map_resized = cv2.resize(probability_map, 
                                        (target_size[1], target_size[0]), 
                                        interpolation=cv2.INTER_LINEAR)
    
    binary_mask_resized = cv2.resize(binary_mask.astype(np.float32), 
                                    (target_size[1], target_size[0]), 
                                    interpolation=cv2.INTER_NEAREST)
    binary_mask_resized = (binary_mask_resized > 0.5).astype(np.float32)
    
    # 2. Imagen original
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Imagen Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 3. Mapa de probabilidades (mostrar el redimensionado)
    prob_im = axes[0, 1].imshow(probability_map_resized, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Mapa de Probabilidades\n(max: {probability_map_resized.max():.3f})', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(prob_im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 4. Máscara binaria (mostrar el redimensionado)
    axes[0, 2].imshow(binary_mask_resized, cmap='gray')
    axes[0, 2].set_title(f'Máscara Binaria (umbral={threshold})\nÁrea: {binary_mask_resized.mean():.2%}', 
                        fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 5. Superposición probabilidad
    # Crear heatmap del tamaño correcto
    heatmap = plt.cm.hot(probability_map_resized)[:, :, :3]  # [H, W, 3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Asegurarse de que ambos tienen el mismo tamaño
    if heatmap.shape[:2] != original_image.shape[:2]:
        # Si aún no coinciden, redimensionar heatmap
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Crear overlay
    overlay_prob = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)
    axes[1, 0].imshow(cv2.cvtColor(overlay_prob, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Superposición: Probabilidades', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 6. Superposición máscara binaria
    overlay_binary = original_image.copy()
    # Crear máscara para resaltar
    mask_colored = np.zeros_like(original_image)
    mask_colored[binary_mask_resized > 0] = [0, 255, 0]  # Verde
    # Aplicar overlay
    overlay_binary = cv2.addWeighted(original_image, 0.7, mask_colored, 0.3, 0)
    axes[1, 1].imshow(cv2.cvtColor(overlay_binary, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Superposición: Máscara Binaria', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 7. Histograma de confianza (usar el probability_map original, no redimensionado)
    axes[1, 2].hist(probability_map.flatten(), bins=50, alpha=0.7, color='purple')
    axes[1, 2].axvline(x=threshold, color='red', linestyle='--', 
                      label=f'Umbral = {threshold}')
    axes[1, 2].set_xlabel('Probabilidad', fontsize=10)
    axes[1, 2].set_ylabel('Frecuencia', fontsize=10)
    axes[1, 2].set_title('Distribución de Confianza', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Segmentación de Lesión Dermatológica\nImagen: {image_name}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Guardar si se especifica
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Resultados guardados en: {save_path}")
    
    plt.show()


def save_segmentation_results(original_image, probability_map, binary_mask, 
                             image_path, output_dir='predictions'):
    """
    Guarda los resultados de segmentación en archivos separados
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    
    # Redimensionar al tamaño original
    target_size = original_image.shape[:2]
    probability_map_resized = cv2.resize(probability_map, 
                                        (target_size[1], target_size[0]), 
                                        interpolation=cv2.INTER_LINEAR)
    
    binary_mask_resized = cv2.resize(binary_mask.astype(np.float32), 
                                    (target_size[1], target_size[0]), 
                                    interpolation=cv2.INTER_NEAREST)
    binary_mask_resized = (binary_mask_resized > 0.5).astype(np.float32)
    
    # 1. Guardar máscara de probabilidades (PNG 16-bit para precisión)
    prob_16bit = (probability_map_resized * 65535).astype(np.uint16)
    prob_path = output_dir / f'{image_name}_probabilities.png'
    cv2.imwrite(str(prob_path), prob_16bit)
    
    # 2. Guardar máscara binaria
    mask_8bit = (binary_mask_resized * 255).astype(np.uint8)
    mask_path = output_dir / f'{image_name}_mask.png'
    cv2.imwrite(str(mask_path), mask_8bit)
    
    # 3. Guardar superposición
    overlay = original_image.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[binary_mask_resized > 0] = [0, 255, 0]  # Verde
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    overlay_path = output_dir / f'{image_name}_overlay.png'
    cv2.imwrite(str(overlay_path), overlay)
    
    # 4. Guardar estadísticas en CSV
    stats = {
        'image': image_name,
        'prob_mean': probability_map.mean(),
        'prob_std': probability_map.std(),
        'prob_max': probability_map.max(),
        'prob_min': probability_map.min(),
        'mask_area': binary_mask.mean(),
        'mask_pixels': binary_mask.sum(),
        'prob_mean_resized': probability_map_resized.mean(),
        'mask_area_resized': binary_mask_resized.mean()
    }
    
    import pandas as pd
    stats_df = pd.DataFrame([stats])
    stats_path = output_dir / f'{image_name}_stats.csv'
    stats_df.to_csv(stats_path, index=False)
    
    print(f"💾 Resultados guardados en {output_dir}/")
    print(f"   • {prob_path.name} (tamaño: {probability_map_resized.shape})")
    print(f"   • {mask_path.name} (tamaño: {binary_mask_resized.shape})")
    print(f"   • {overlay_path.name}")
    print(f"   • {stats_path.name}")

def main():
    parser = argparse.ArgumentParser(
        description='Segmentar lesiones dermatológicas en una imagen usando modelos entrenados'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--image', type=str, required=True,
                       help='Ruta a la imagen a segmentar (.jpg, .png, etc.)')
    parser.add_argument('--model_type', type=str, 
                       choices=['unet', 'attention', 'residual', 'auto'],
                       default='auto', help='Tipo de modelo (auto para autodetectar)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Umbral para binarización (0-1)')
    parser.add_argument('--device', type=str, 
                       choices=['cpu', 'cuda', 'auto'],
                       default='auto', help='Dispositivo para inferencia')
    parser.add_argument('--save', action='store_true',
                       help='Guardar resultados en disco')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Directorio para guardar resultados')
    parser.add_argument('--no_display', action='store_true',
                       help='No mostrar visualización (solo guardar)')
    
    args = parser.parse_args()
    
    print("🔍 SEGMENTADOR DE LESIONES DERMATOLÓGICAS")
    print("="*50)
    
    # Configurar dispositivo
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"📱 Dispositivo: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Cargar modelo
    print(f"\n🧠 Cargando modelo: {args.model}")
    model = load_model(args.model, args.model_type, device)
    
    # 2. Preprocesar imagen
    print(f"\n📸 Procesando imagen: {args.image}")
    try:
        image_tensor, original_image = preprocess_image(args.image)
        print(f"   Tamaño original: {original_image.shape[:2]}")
        print(f"   Tamaño procesado: {image_tensor.shape[1:]}")  # H, W
    except Exception as e:
        print(f"❌ Error procesando imagen: {e}")
        sys.exit(1)
    
    # 3. Realizar predicción
    print(f"\n🤖 Realizando segmentación...")
    probability_map, binary_mask = predict_mask(
        model, image_tensor, device, args.threshold
    )
    
    # Calcular estadísticas
    print(f"📊 Resultados:")
    print(f"   • Probabilidad máxima: {probability_map.max():.3f}")
    print(f"   • Probabilidad media: {probability_map.mean():.3f}")
    print(f"   • Área segmentada: {binary_mask.mean():.2%}")
    print(f"   • Píxeles segmentados: {binary_mask.sum():.0f}")
    
    # 4. Visualizar resultados
    if not args.no_display:
        print(f"\n🎨 Generando visualización...")
        image_name = Path(args.image).name
        save_path = Path(args.output_dir) / f'{Path(args.image).stem}_results.png' if args.save else None
        
        visualize_results(
            original_image=original_image,
            probability_map=probability_map,
            binary_mask=binary_mask,
            image_name=image_name,
            threshold=args.threshold,
            save_path=save_path
        )
    
    # 5. Guardar resultados si se solicita
    if args.save:
        print(f"\n💾 Guardando resultados...")
        save_segmentation_results(
            original_image=original_image,
            probability_map=probability_map,
            binary_mask=binary_mask,
            image_path=args.image,
            output_dir=args.output_dir
        )
    
    print(f"\n✅ Segmentación completada exitosamente!")
    
    # Sugerencia para ajustar umbral
    if 0.45 < probability_map.mean() < 0.55:
        print(f"\n💡 Sugerencia: La probabilidad media ({probability_map.mean():.3f}) está cerca del umbral.")
        print(f"   Considera probar con --threshold {probability_map.mean():.2f}")

if __name__ == "__main__":
    main()