#!/usr/bin/env python3
"""
Script principal de entrenamiento
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml
import argparse
from datetime import datetime

from src.data_preparation.data_loader import ISICDataLoader
from src.models import UNet, AttentionUNet, ResidualUNet
from src.training.trainer import SegmentationTrainer

def load_config(config_path: str) -> dict:
    """Carga configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo de segmentación')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Ruta al archivo de configuración')
    parser.add_argument('--model', type=str, choices=['unet', 'attention', 'residual'],
                       default='unet', help='Arquitectura del modelo')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Tamaño del batch')
    parser.add_argument('--experiment', type=str,
                       help='Nombre del experimento')
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Inicializar secciones si no existen
    if 'data' not in config:
        config['data'] = {}
    if 'models' not in config:
        config['models'] = {}
    if 'training' not in config:
        config['training'] = {}
    
    # Sobrescribir con argumentos de línea de comandos
    config['training']['epochs'] = args.epochs
    config['data']['batch_size'] = args.batch_size
    
    # Configurar experimento
    if args.experiment:
        exp_name = args.experiment
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{args.model}_{timestamp}"
    
    config['training']['experiment_name'] = exp_name
    
    print("INICIANDO ENTRENAMIENTO")
    print("="*50)
    print(f"Configuración:")
    print(f"Modelo: {args.model}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Experimento: {exp_name}")
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Cargar datos
    print("\nCargando datos...")
    data_loader = ISICDataLoader(
        base_path=config['data']['base_path'],
        batch_size=config['data']['batch_size'],
        target_size=tuple(config['data']['target_size']),
        num_workers=config['data'].get('num_workers', 4),
        use_class_balancing=config['data'].get('use_class_balancing', True)
    )
    
    data_loader.create_datasets(
        use_processed=config['data'].get('use_processed', True),
        augment_train=config['data'].get('augment_train', True)
    )
    data_loader.create_dataloaders()
    
    # Crear modelo
    print(f"\nCreando modelo {args.model}...")
    
    model_config = config['models'][args.model]
    
    if args.model == 'unet':
        model = UNet(
            in_channels=3,
            out_channels=1,
            features=model_config.get('features', [64, 128, 256, 512])
        )
    elif args.model == 'attention':
        model = AttentionUNet(
            in_channels=3,
            out_channels=1,
            features=model_config.get('features', [64, 128, 256, 512])
        )
    elif args.model == 'residual':
        model = ResidualUNet(
            in_channels=3,
            out_channels=1,
            features=model_config.get('features', [64, 128, 256, 512])
        )
    
    model.to(device)
    
    # Crear trainer
    print("\nConfigurando trainer...")
    trainer = SegmentationTrainer(
        model=model,
        train_loader=data_loader.train_loader,
        val_loader=data_loader.val_loader,
        device=device,
        config=config['training']
    )
    
    # Entrenar
    print("\n" + "="*50)
    history = trainer.train(epochs=config['training']['epochs'])
    
    # Visualizar resultados
    print("\nVisualizando resultados...")
    trainer.visualize_results(num_samples=min(6, config['data']['batch_size']))
    
    # Guardar historial
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_path = trainer.exp_dir / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    
    # Guardar configuración
    config_path = trainer.exp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nEntrenamiento completado!")
    print(f"Resultados guardados en: {trainer.exp_dir}")
    print(f"Configuración: {config_path}")
    print(f"Historial: {history_path}")
    print(f"Mejor val_dice: {trainer.best_val_dice:.4f}")
    print(f"Mejor val_loss: {trainer.best_val_loss:.4f}")

if __name__ == "__main__":
    main()