"""
U-Net++ (Nested U-Net) para segmentación de imágenes médicas

Referencia: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
https://arxiv.org/abs/1807.10165

Características principales:
- Conexiones densas anidadas entre encoder y decoder
- Mejora la fusión de características multi-escala
- Supervisión profunda opcional para mejor entrenamiento
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Bloque convolucional doble con BatchNorm y ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) con conexiones densas anidadas
    
    La arquitectura forma una estructura de red densa donde cada nodo
    en el decoder está conectado con todos los nodos anteriores en su
    nivel y con el nodo correspondiente del encoder.
    
    Args:
        in_channels: Número de canales de entrada (default: 3 para RGB)
        out_channels: Número de canales de salida (default: 1 para segmentación binaria)
        features: Lista con número de filtros por nivel (default: [64, 128, 256, 512])
        deep_supervision: Si True, retorna salidas de múltiples niveles (default: False)
    """
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], 
                 deep_supervision=False):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.n_levels = len(features)
        
        # Crear una matriz de bloques convolucionales
        # x_{i,j} donde i es el nivel (profundidad) y j es la posición en ese nivel
        self.conv_blocks = nn.ModuleDict()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.ModuleDict()
        
        # Nivel 0 (encoder) - x_{0,0}, x_{1,0}, x_{2,0}, x_{3,0}
        for i, feature in enumerate(features):
            if i == 0:
                self.conv_blocks[f'x_{i}_0'] = ConvBlock(in_channels, feature)
            else:
                self.conv_blocks[f'x_{i}_0'] = ConvBlock(features[i-1], feature)
        
        # Bottleneck - x_{4,0}
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Crear upsampling layers
        # Upsample desde bottleneck
        self.up['up_4_0'] = nn.ConvTranspose2d(features[-1] * 2, features[-1], 
                                                kernel_size=2, stride=2)
        
        # Upsampling para cada nivel del decoder
        for i in range(self.n_levels):
            for j in range(1, self.n_levels - i + 1):
                self.up[f'up_{i}_{j}'] = nn.ConvTranspose2d(features[i], features[i-1] if i > 0 else features[0], 
                                                            kernel_size=2, stride=2)
        
        # Bloques del decoder anidado
        # x_{i,j} para j > 0
        for i in range(self.n_levels):
            for j in range(1, self.n_levels - i + 1):
                # Número de canales de entrada = feature del nivel + (j conexiones skip del mismo nivel)
                # + feature del nivel inferior (upsampled)
                if i == self.n_levels - 1:
                    # Último nivel del encoder, recibe del bottleneck
                    in_ch = features[i] * (j) + features[i]  # j skips + 1 upsampled
                else:
                    # Niveles intermedios
                    in_ch = features[i] * (j) + features[i]  # j skips + 1 upsampled
                
                self.conv_blocks[f'x_{i}_{j}'] = ConvBlock(in_ch, features[i])
        
        # Capas de salida
        if deep_supervision:
            self.output_layers = nn.ModuleList([
                nn.Conv2d(features[0], out_channels, kernel_size=1)
                for _ in range(self.n_levels)
            ])
        else:
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass de U-Net++
        
        La estructura de conexiones es:
        
        x_0_0 → x_0_1 → x_0_2 → x_0_3 → x_0_4 (outputs)
          ↓   ↗   ↓   ↗   ↓   ↗   ↓   ↗
        x_1_0 → x_1_1 → x_1_2 → x_1_3
          ↓   ↗   ↓   ↗   ↓   ↗
        x_2_0 → x_2_1 → x_2_2
          ↓   ↗   ↓   ↗
        x_3_0 → x_3_1
          ↓   ↗
        bottleneck
        """
        
        # Diccionario para almacenar todas las features
        features_dict = {}
        
        # Encoder path (columna j=0)
        for i in range(self.n_levels):
            if i == 0:
                features_dict[f'x_{i}_0'] = self.conv_blocks[f'x_{i}_0'](x)
            else:
                features_dict[f'x_{i}_0'] = self.conv_blocks[f'x_{i}_0'](
                    self.pool(features_dict[f'x_{i-1}_0'])
                )
        
        # Bottleneck
        bottleneck_out = self.bottleneck(self.pool(features_dict[f'x_{self.n_levels-1}_0']))
        
        # Decoder path (columnas j=1,2,3,...)
        # Procesamos diagonal por diagonal
        for j in range(1, self.n_levels + 1):
            for i in range(self.n_levels - j, -1, -1):
                # Recoger skip connections del mismo nivel (todas las x_{i, 0..j-1})
                skip_features = [features_dict[f'x_{i}_{k}'] for k in range(j)]
                
                # Obtener features upsampled del nivel inferior
                if i == self.n_levels - 1 and j == 1:
                    # Primera conexión desde bottleneck
                    up_feature = self.up['up_4_0'](bottleneck_out)
                else:
                    # Conexión desde el nivel inferior
                    up_feature = self._upsample(
                        features_dict[f'x_{i+1}_{j-1}'],
                        features_dict[f'x_{i}_0']
                    )
                
                # Ajustar tamaño si es necesario
                target_size = skip_features[0].shape[2:]
                if up_feature.shape[2:] != target_size:
                    up_feature = F.interpolate(up_feature, size=target_size,
                                              mode='bilinear', align_corners=False)
                
                # Concatenar skip connections + upsampled features
                concat_features = torch.cat(skip_features + [up_feature], dim=1)
                
                # Aplicar bloque convolucional
                features_dict[f'x_{i}_{j}'] = self.conv_blocks[f'x_{i}_{j}'](concat_features)
        
        # Output
        if self.deep_supervision and self.training:
            outputs = []
            for j in range(1, self.n_levels + 1):
                out = self.output_layers[j-1](features_dict[f'x_0_{j}'])
                outputs.append(out)
            return outputs
        else:
            if self.deep_supervision:
                # Durante inferencia con deep supervision, usar la última salida
                return self.output_layers[-1](features_dict[f'x_0_{self.n_levels}'])
            else:
                return self.final_conv(features_dict[f'x_0_{self.n_levels}'])
    
    def _upsample(self, x, target):
        """Upsample x para que coincida con el tamaño de target"""
        return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)


class UNetPlusPlusSimplified(nn.Module):
    """
    Versión simplificada y más eficiente de U-Net++ para 4 niveles
    
    Esta implementación es más directa y sigue exactamente la estructura
    del paper original para L=4 niveles.
    """
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
                 deep_supervision=False):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        f = features  # Alias para código más legible
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder (backbone)
        self.conv0_0 = ConvBlock(in_channels, f[0])
        self.conv1_0 = ConvBlock(f[0], f[1])
        self.conv2_0 = ConvBlock(f[1], f[2])
        self.conv3_0 = ConvBlock(f[2], f[3])
        self.conv4_0 = ConvBlock(f[3], f[3] * 2)  # Bottleneck
        
        # Decoder nivel 0 (más superficial)
        self.up0_1 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.conv0_1 = ConvBlock(f[0] * 2, f[0])
        
        self.up0_2 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.conv0_2 = ConvBlock(f[0] * 3, f[0])
        
        self.up0_3 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.conv0_3 = ConvBlock(f[0] * 4, f[0])
        
        self.up0_4 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.conv0_4 = ConvBlock(f[0] * 5, f[0])
        
        # Decoder nivel 1
        self.up1_1 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.conv1_1 = ConvBlock(f[1] * 2, f[1])
        
        self.up1_2 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.conv1_2 = ConvBlock(f[1] * 3, f[1])
        
        self.up1_3 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.conv1_3 = ConvBlock(f[1] * 4, f[1])
        
        # Decoder nivel 2
        self.up2_1 = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.conv2_1 = ConvBlock(f[2] * 2, f[2])
        
        self.up2_2 = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.conv2_2 = ConvBlock(f[2] * 3, f[2])
        
        # Decoder nivel 3
        self.up3_1 = nn.ConvTranspose2d(f[3] * 2, f[3], 2, 2)
        self.conv3_1 = ConvBlock(f[3] * 2, f[3])
        
        # Output layers
        if deep_supervision:
            self.output1 = nn.Conv2d(f[0], out_channels, 1)
            self.output2 = nn.Conv2d(f[0], out_channels, 1)
            self.output3 = nn.Conv2d(f[0], out_channels, 1)
            self.output4 = nn.Conv2d(f[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f[0], out_channels, 1)
    
    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        # Decoder - Primera diagonal
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(self.up3_1(x4_0), x3_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(self.up2_1(x3_0), x2_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(self.up1_1(x2_0), x1_0)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(self.up0_1(x1_0), x0_0)], 1))
        
        # Decoder - Segunda diagonal
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(self.up2_2(x3_1), x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(self.up1_2(x2_1), x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(self.up0_2(x1_1), x0_0)], 1))
        
        # Decoder - Tercera diagonal
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(self.up1_3(x2_2), x1_0)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(self.up0_3(x1_2), x0_0)], 1))
        
        # Decoder - Cuarta diagonal (final)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(self.up0_4(x1_3), x0_0)], 1))
        
        # Output
        if self.deep_supervision and self.training:
            out1 = self.output1(x0_1)
            out2 = self.output2(x0_2)
            out3 = self.output3(x0_3)
            out4 = self.output4(x0_4)
            return [out1, out2, out3, out4]
        else:
            if self.deep_supervision:
                return self.output4(x0_4)
            else:
                return self.final_conv(x0_4)
    
    def _up(self, x, target):
        """Ajusta el tamaño de x para que coincida con target"""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x
