#!/usr/bin/env python3
"""
Prueba todas las arquitecturas de modelos
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.models import UNet, AttentionUNet, ResidualUNet

def test_all_models():
    print("🧠 PROBANDO TODAS LAS ARQUITECTURAS")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Dispositivo: {device}")
    
    # Configuración común
    batch_size = 2
    in_channels = 3
    out_channels = 1
    features = [64, 128, 256, 512]
    input_size = (batch_size, in_channels, 256, 256)
    
    # Crear input de prueba
    test_input = torch.randn(*input_size).to(device)
    
    # Probar cada modelo
    models = [
        ("U-Net Básica", UNet(in_channels, out_channels, features)),
        ("Attention U-Net", AttentionUNet(in_channels, out_channels, features)),
        ("Residual U-Net", ResidualUNet(in_channels, out_channels, features))
    ]
    
    for name, model in models:
        print(f"\n🔍 {name}:")
        model.to(device)
        model.eval()
        
        # Contar parámetros
        params = sum(p.numel() for p in model.parameters())
        
        # Forward pass
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parámetros: {params:,}")
        print(f"  Tamaño en MB: {params * 4 / (1024**2):.2f}")
        
        # Verificar rango de salida
        print(f"  Rango output: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n✅ Todas las arquitecturas probadas exitosamente!")

if __name__ == "__main__":
    test_all_models()