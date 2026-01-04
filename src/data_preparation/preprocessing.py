import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

class MedicalImagePreprocessor:
    """Preprocesador para imágenes médicas ISIC 2018"""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        
    def load_image_mask(self, image_path: Path, mask_path: Optional[Path] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Carga imagen y máscara (si existe)
        
        Args:
            image_path: Ruta a la imagen .jpg
            mask_path: Ruta a la máscara .png (opcional)
            
        Returns:
            Tupla (imagen, máscara) o (imagen, None)
        """
        # Cargar imagen (convertir a RGB)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Cargar máscara si existe
        mask = None
        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # Asegurar binaria (0 o 255)
            mask = (mask > 0).astype(np.uint8) * 255
        
        return image, mask
    
    def resize_pair(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Redimensiona imagen y máscara manteniendo relación de aspecto
        
        Args:
            image: Imagen RGB
            mask: Máscara binaria (opcional)
            
        Returns:
            Tupla (imagen_redim, máscara_redim)
        """
        # Redimensionar imagen
        image_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Redimensionar máscara si existe
        mask_resized = None
        if mask is not None:
            mask_resized = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            # Mantener binarizada después del resize
            mask_resized = (mask_resized > 127).astype(np.uint8) * 255
        
        return image_resized, mask_resized
    
    def normalize_image(self, image: np.ndarray, method: str = 'zero_one') -> np.ndarray:
        """
        Normaliza la imagen
        
        Args:
            image: Imagen en formato uint8 [0, 255]
            method: 'zero_one' (0-1), 'imagenet' (ImageNet stats), 'custom'
            
        Returns:
            Imagen normalizada
        """
        if method == 'zero_one':
            return image.astype(np.float32) / 255.0
            
        elif method == 'imagenet':
            # Estadísticas de ImageNet
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_normalized = image.astype(np.float32) / 255.0
            return (image_normalized - mean) / std
            
        elif method == 'custom':
            # Basado en nuestro EDA: media ~149, normalizar a [-1, 1]
            image_normalized = image.astype(np.float32)
            return (image_normalized - 149.0) / (255.0 / 2)
            
        else:
            raise ValueError(f"Método {method} no reconocido")
    
    def augment_pair(self, image: np.ndarray, mask: np.ndarray, 
                    augmentation_strength: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica aumentación de datos a par imagen-máscara
        
        Args:
            image: Imagen normalizada
            mask: Máscara binaria
            augmentation_strength: 'light', 'medium', 'strong'
            
        Returns:
            Tupla (imagen_aug, máscara_aug)
        """
        # Definir transformaciones según intensidad
        if augmentation_strength == 'light':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
            ])
            
        elif augmentation_strength == 'medium':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.3),
            ])
            
        elif augmentation_strength == 'strong':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.GaussianBlur(blur_limit=5, p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            ])
        
        else:
            transform = A.Compose([])
        
        # Aplicar transformaciones
        augmented = transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    
    def process_single_pair(self, image_path: Path, mask_path: Optional[Path] = None,
                           normalize: bool = True, augment: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Pipeline completo para un par imagen-máscara
        
        Args:
            image_path: Ruta a imagen
            mask_path: Ruta a máscara (opcional)
            normalize: Si aplicar normalización
            augment: Si aplicar aumentación (solo si hay máscara)
            
        Returns:
            Tupla (imagen_procesada, máscara_procesada)
        """
        # 1. Cargar
        image, mask = self.load_image_mask(image_path, mask_path)
        
        # 2. Redimensionar
        image_resized, mask_resized = self.resize_pair(image, mask)
        
        # 3. Normalizar imagen
        if normalize:
            image_normalized = self.normalize_image(image_resized, method='zero_one')
        else:
            image_normalized = image_resized.astype(np.float32)
        
        # 4. Aumentar (solo si hay máscara y se solicita)
        if augment and mask_resized is not None:
            image_aug, mask_aug = self.augment_pair(image_normalized, mask_resized)
            return image_aug, mask_aug
        
        # Convertir máscara a float si existe
        if mask_resized is not None:
            mask_float = mask_resized.astype(np.float32) / 255.0
            return image_normalized, mask_float
        
        return image_normalized, None

    def create_torch_transforms(self, phase: str = 'train') -> A.Compose:
        """
        Crea transformaciones para PyTorch
        
        Args:
            phase: 'train', 'val', o 'test'
            
        Returns:
            Transformaciones de albumentations
        """
        # Transformaciones base para todas las fases
        base_transforms = [
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        
        if phase == 'train':
            # Añadir aumentación para entrenamiento
            transforms_list = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        else:
            # Solo resize y normalización para val/test
            transforms_list = base_transforms
        
        return A.Compose(transforms_list)