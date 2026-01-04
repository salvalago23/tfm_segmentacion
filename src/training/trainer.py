import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import warnings
warnings.filterwarnings('ignore')

from ..models.metrics import SegmentationMetrics
from ..models.losses import CombinedLoss

class SegmentationTrainer:
    """
    Trainer profesional para segmentación semántica
    Soporta múltiples modelos, pérdidas y métricas
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 config: Dict[str, Any]):
        """
        Args:
            model: Modelo de segmentación
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            device: Dispositivo (cuda/cpu)
            config: Configuración del entrenamiento
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Optimizador
        self.optimizer = self._create_optimizer()
        
        # Función de pérdida
        self.criterion = self._create_criterion()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Métricas
        self.train_metrics = SegmentationMetrics()
        self.val_metrics = SegmentationMetrics()
        
        # Historial
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rate': []
        }
        
        # Mejores pesos
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        
        # Crear directorios
        self._create_directories()
        
        print(f"✅ Trainer inicializado para {model.__class__.__name__}")
        print(f"   Device: {device}")
        print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Crea el optimizador según configuración"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, 
                           weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizador no soportado: {optimizer_name}")
    
    def _create_criterion(self) -> nn.Module:
        """Crea la función de pérdida"""
        loss_config = self.config.get('loss', {
            'name': 'combined',
            'losses': ['bce', 'dice'],
            'weights': [0.5, 0.5]
        })
        
        loss_name = loss_config.get('name', 'combined')
        
        if loss_name == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_name == 'dice':
            from ..models.losses import DiceLoss
            return DiceLoss()
        elif loss_name == 'combined':
            return CombinedLoss(
                losses=loss_config.get('losses', ['bce', 'dice']),
                weights=loss_config.get('weights', [0.5, 0.5])
            )
        else:
            raise ValueError(f"Pérdida no soportada: {loss_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Crea el scheduler de learning rate"""
        scheduler_name = self.config.get('scheduler', 'reduce_on_plateau')
        
        if scheduler_name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                # verbose=True  # <- ELIMINAR ESTA LÍNEA
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 50)
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            return None
    
    def _create_directories(self):
        """Crea directorios para guardar resultados"""
        base_dir = Path(self.config.get('output_dir', 'experiments'))
        exp_name = self.config.get('experiment_name', 'exp_01')
        
        self.exp_dir = base_dir / exp_name
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.logs_dir = self.exp_dir / 'logs'
        self.results_dir = self.exp_dir / 'results'
        
        for dir_path in [self.exp_dir, self.checkpoint_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Entrena por una época"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc='Entrenamiento', leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Mover datos al dispositivo
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calcular pérdida
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (opcional)
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Actualizar métricas
            epoch_loss += loss.item()
            self.train_metrics.update(outputs, masks)
            
            # Actualizar progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'dice': self.train_metrics.compute()['dice']
            })
        
        # Calcular métricas finales
        avg_loss = epoch_loss / num_batches
        metrics = self.train_metrics.compute()
        
        return avg_loss, metrics
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Valida por una época"""
        self.model.eval()
        self.val_metrics.reset()
        
        epoch_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validación', leave=False)
            
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calcular pérdida
                loss = self.criterion(outputs, masks)
                
                # Actualizar métricas
                epoch_loss += loss.item()
                self.val_metrics.update(outputs, masks)
                
                # Actualizar progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'dice': self.val_metrics.compute()['dice']
                })
        
        # Calcular métricas finales
        avg_loss = epoch_loss / num_batches
        metrics = self.val_metrics.compute()
        
        return avg_loss, metrics
    
    def train(self, epochs: int) -> Dict[str, List[float]]:
        """
        Entrena el modelo por el número especificado de épocas
        
        Args:
            epochs: Número de épocas
            
        Returns:
            Historial de entrenamiento
        """
        print(f"\n🚀 Comenzando entrenamiento por {epochs} épocas")
        print(f"   Modelo: {self.model.__class__.__name__}")
        print(f"   Pérdida: {self.criterion.__class__.__name__}")
        print(f"   Optimizador: {self.optimizer.__class__.__name__}")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"\n📊 Época {epoch}/{epochs}")
            
            # Entrenar
            train_loss, train_metrics = self.train_epoch()
            
            # Validar
            val_loss, val_metrics = self.validate_epoch()
            
            # Actualizar scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                    # Mostrar mensaje manualmente si el LR cambió
                    old_lr = self.history['learning_rate'][-1] if self.history['learning_rate'] else self.optimizer.param_groups[0]['lr']
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr < old_lr:
                        print(f"   🔽 LR reducido: {old_lr:.2e} → {new_lr:.2e}")
                else:
                    self.scheduler.step()
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Mostrar resultados
            print(f"   Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
            print(f"   Val   - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Guardar checkpoint si es mejor
            self._save_checkpoint(epoch, val_loss, val_metrics['dice'])
            
            # Early stopping (si se activa)
            if self._check_early_stopping():
                print("⏹️  Early stopping activado")
                break
        
        # Entrenamiento completado
        training_time = time.time() - start_time
        print(f"\n✅ Entrenamiento completado en {training_time:.2f} segundos")
        print(f"   Mejor val_loss: {self.best_val_loss:.4f}")
        print(f"   Mejor val_dice: {self.best_val_dice:.4f}")
        
        # Guardar modelo final
        self._save_final_model()
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, val_loss: float, val_dice: float):
        """Guarda checkpoint si el modelo mejora"""
        
        # Verificar si es mejor en pérdida
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_path = self.checkpoint_dir / f'best_loss_epoch_{epoch}.pth'
            self._save_model(checkpoint_path, epoch, val_loss, val_dice)
            print(f"   💾 Nuevo mejor val_loss: {val_loss:.4f}")
        
        # Verificar si es mejor en dice
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
            checkpoint_path = self.checkpoint_dir / f'best_dice_epoch_{epoch}.pth'
            self._save_model(checkpoint_path, epoch, val_loss, val_dice)
            print(f"   💾 Nuevo mejor val_dice: {val_dice:.4f}")
        
        # Guardar checkpoint periódico
        if epoch % self.config.get('save_every', 10) == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            self._save_model(checkpoint_path, epoch, val_loss, val_dice)
    
    def _save_model(self, path: Path, epoch: int, val_loss: float, val_dice: float):
        """Guarda el estado del modelo de forma compatible"""
        # Crear diccionario de checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_dice': val_dice,
            'config': self.config,
            'history': self.history
        }
        
        # Añadir scheduler si existe
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Guardar con formato compatible
        try:
            # Usar _use_new_zipfile_serialization=False para compatibilidad
            torch.save(checkpoint, str(path), _use_new_zipfile_serialization=False)
        except TypeError:
            # Si el argumento no es soportado, guardar normalmente
            torch.save(checkpoint, str(path))

    def _save_final_model(self):
        """Guarda el modelo final de forma compatible"""
        final_path = self.exp_dir / 'final_model.pth'
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'training_info': {
                'model_class': self.model.__class__.__name__,
                'criterion_class': self.criterion.__class__.__name__,
                'optimizer_class': self.optimizer.__class__.__name__,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        try:
            torch.save(checkpoint, str(final_path), _use_new_zipfile_serialization=False)
        except TypeError:
            torch.save(checkpoint, str(final_path))
        
        print(f"💾 Modelo final guardado en: {final_path}")
        
    def _check_early_stopping(self) -> bool:
        """Verifica condiciones de early stopping"""
        if not self.config.get('early_stopping', False):
            return False
        
        patience = self.config.get('patience', 10)
        
        if len(self.history['val_loss']) < patience:
            return False
        
        # Verificar si no ha mejorado en 'patience' épocas
        recent_losses = self.history['val_loss'][-patience:]
        best_recent = min(recent_losses)
        
        return best_recent >= self.best_val_loss
    
    def visualize_results(self, num_samples: int = 4):
        """Visualiza resultados del modelo"""
        import matplotlib.pyplot as plt
        
        self.model.eval()
        
        # Obtener un batch de validación
        batch = next(iter(self.val_loader))
        
        # Asegurar que num_samples no exceda el batch_size
        batch_size = batch['image'].size(0)
        num_samples = min(num_samples, batch_size)
        
        if num_samples <= 0:
            print("⚠️  No hay muestras para visualizar")
            return
        
        images = batch['image'].to(self.device)[:num_samples]
        true_masks = batch['mask'].to(self.device)[:num_samples]
        image_ids = batch['image_id'][:num_samples]
        
        # Predicciones
        with torch.no_grad():
            preds = self.model(images)
            preds_sigmoid = torch.sigmoid(preds)
            preds_binary = (preds_sigmoid > 0.5).float()
        
        # Configurar figura
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Resultados - {self.model.__class__.__name__}', fontsize=16, y=1.02)
        
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
            pred_prob = preds_sigmoid[i, 0].cpu().numpy()
            pred_mask = preds_binary[i, 0].cpu().numpy()
            
            # Calcular métricas para esta muestra
            from ..models.metrics import dice_score, iou_score
            dice = dice_score(preds[i:i+1], true_masks[i:i+1])
            iou = iou_score(preds[i:i+1], true_masks[i:i+1])
            
            # Imagen original
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Imagen\n{image_ids[i][:10]}...")
            axes[i, 0].axis('off')
            
            # Máscara verdadera
            axes[i, 1].imshow(true_mask, cmap='gray')
            axes[i, 1].set_title(f"Verdadero\nCov: {true_mask.mean():.1%}")
            axes[i, 1].axis('off')
            
            # Predicción probabilística
            axes[i, 2].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title(f"Predicción\nDice: {dice:.3f}")
            axes[i, 2].axis('off')
            
            # Predicción binaria
            axes[i, 3].imshow(pred_mask, cmap='gray')
            axes[i, 3].set_title(f"Binaria\nIoU: {iou:.3f}")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        # Guardar figura
        save_path = self.results_dir / 'validation_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Resultados guardados en: {save_path}")