#!/usr/bin/env python3
"""
Script para evaluar modelos entrenados - VERSIÓN CORREGIDA
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data_preparation.data_loader import ISICDataLoader
from src.models import UNet, AttentionUNet, ResidualUNet
from src.models.metrics import SegmentationMetrics

def safe_load_checkpoint(model_path, device):
    """Carga un checkpoint de forma segura"""
    try:
        # PyTorch 2.6+ necesita weights_only=False para algunos checkpoints
        return torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error con weights_only=False: {e}")
        # Intentar método alternativo
        try:
            return torch.load(model_path, map_location=device)
        except Exception as e2:
            print(f"Error carga alternativa: {e2}")
            raise

def load_model(model_path, model_type='unet', device='cuda'):
    """Carga un modelo entrenado"""
    checkpoint = safe_load_checkpoint(model_path, device)
    
    # Extraer configuración
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Configuración por defecto si no está en el checkpoint
        config = {
            'data': {
                'base_path': 'data',
                'target_size': [256, 256],
                'use_processed': True
            }
        }
    
    # Determinar tipo de modelo
    if model_type == 'auto':
        model_type = 'unet'  # Por defecto
    
    # Crear modelo
    if model_type == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif model_type == 'attention':
        model = AttentionUNet(in_channels=3, out_channels=1)
    elif model_type == 'residual':
        model = ResidualUNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    # Cargar pesos
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Asumir que el checkpoint ES el state_dict
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config

def evaluate_model(model, data_loader, device, threshold=0.5):
    """Evalúa el modelo en un DataLoader"""
    metrics = SegmentationMetrics(threshold=threshold)
    
    all_predictions = []
    all_targets = []
    image_ids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluando'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Predicción
            outputs = model(images)
            
            # Actualizar métricas
            metrics.update(outputs, masks)
            
            # Guardar para análisis detallado
            all_predictions.append(torch.sigmoid(outputs).cpu())
            all_targets.append(masks.cpu())
            image_ids.extend(batch['image_id'])
    
    return metrics.compute(), all_predictions, all_targets, image_ids

def visualize_predictions(model, data_loader, device, num_samples=4, save_dir=None):
    """Visualiza predicciones del modelo"""
    model.eval()
    
    # Obtener batch, asegurando suficientes muestras
    batch = next(iter(data_loader))
    batch_size = batch['image'].size(0)
    num_samples = min(num_samples, batch_size)
    
    images = batch['image'].to(device)[:num_samples]
    true_masks = batch['mask'].to(device)[:num_samples]
    batch_ids = batch['image_id'][:num_samples]
    
    # Predicciones
    with torch.no_grad():
        outputs = model(images)
        preds_probs = torch.sigmoid(outputs)
        preds_binary = (preds_probs > 0.5).float()
    
    # Crear figura
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Desnormalizar imagen
        img_np = images[i].cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Máscaras
        true_mask = true_masks[i, 0].cpu().numpy()
        pred_prob = preds_probs[i, 0].cpu().numpy()
        pred_binary = preds_binary[i, 0].cpu().numpy()
        
        # Calcular métricas
        from src.models.metrics import dice_score, iou_score
        dice = dice_score(outputs[i:i+1], true_masks[i:i+1])
        iou = iou_score(outputs[i:i+1], true_masks[i:i+1])
        
        # Imagen original
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Imagen\n{batch_ids[i][:12]}...", fontsize=10)
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title(f"Verdadero\nCov: {true_mask.mean():.1%}", fontsize=10)
        axes[i, 1].axis('off')
        
        # Probabilidad
        im_prob = axes[i, 2].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Probabilidad\nMax: {pred_prob.max():.2f}", fontsize=10)
        axes[i, 2].axis('off')
        plt.colorbar(im_prob, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Predicción binaria
        axes[i, 3].imshow(pred_binary, cmap='gray')
        axes[i, 3].set_title(f"Predicción\nDice: {dice:.3f}", fontsize=10)
        axes[i, 3].axis('off')
        
        # Diferencia
        diff = np.abs(pred_binary - true_mask)
        axes[i, 4].imshow(diff, cmap='Reds', vmin=0, vmax=1)
        axes[i, 4].set_title(f"Diferencia\nIoU: {iou:.3f}", fontsize=10)
        axes[i, 4].axis('off')
    
    plt.suptitle(f'Evaluación - {model.__class__.__name__}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Guardar si se especifica directorio
    if save_dir:
        save_path = Path(save_dir) / 'evaluation_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluar modelo de segmentación')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--model_type', type=str, 
                       choices=['unet', 'attention', 'residual', 'auto'],
                       default='auto', help='Tipo de modelo')
    parser.add_argument('--data_split', type=str, choices=['val', 'test'],
                       default='val', help='Split a evaluar')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Tamaño del batch (ajustar según GPU)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Umbral para binarización')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualizar predicciones')
    parser.add_argument('--save_results', type=str,
                       help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Cargar modelo
    print(f"\nCargando modelo: {args.model_path}")
    model, config = load_model(args.model_path, args.model_type, device)
    
    # Cargar datos
    print(f"\nCargando datos {args.data_split}...")
    data_loader = ISICDataLoader(
        base_path=config.get('data', {}).get('base_path', 'data'),
        batch_size=args.batch_size,
        target_size=tuple(config.get('data', {}).get('target_size', [256, 256])),
        num_workers=2,
        use_class_balancing=False  # No balancear en evaluación
    )
    
    data_loader.create_datasets(
        use_processed=config.get('data', {}).get('use_processed', True),
        augment_train=False  # Sin aumentación en evaluación
    )
    
    if args.data_split == 'val':
        data_loader.create_dataloaders(shuffle_train=False)
        eval_loader = data_loader.val_loader
    else:  # test
        data_loader.create_dataloaders(shuffle_train=False)
        if data_loader.test_loader:
            eval_loader = data_loader.test_loader
        else:
            print("Test dataset no disponible, usando validation")
            eval_loader = data_loader.val_loader
    
    # Evaluar
    print(f"\nEvaluando modelo...")
    metrics, predictions, targets, image_ids = evaluate_model(
        model, eval_loader, device, args.threshold
    )
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*50)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name:>12}: {value:.4f}")
    
    # Guardar resultados
    if args.save_results:
        save_dir = Path(args.save_results)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar métricas
        metrics_df = pd.DataFrame([metrics])
        metrics_path = save_dir / 'metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nMétricas guardadas en: {metrics_path}")
        
        # Guardar predicciones detalladas
        predictions_flat = torch.cat(predictions, dim=0).numpy()
        targets_flat = torch.cat(targets, dim=0).numpy()
        
        np.save(save_dir / 'predictions.npy', predictions_flat)
        np.save(save_dir / 'targets.npy', targets_flat)
    
    # Visualizar predicciones
    if args.visualize:
        print(f"\nVisualizando predicciones...")
        visualize_predictions(
            model, eval_loader, device, 
            num_samples=min(4, args.batch_size),  # Asegurar que no exceda batch_size
            save_dir=args.save_results
        )
    
    print(f"\nEvaluación completada!")

if __name__ == "__main__":
    main()