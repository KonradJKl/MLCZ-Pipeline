import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import MetricCollection, JaccardIndex, Accuracy, F1Score

from typing import Any, Dict, List, Tuple


class BaseModel(L.LightningModule):
    def __init__(self, args, datamodule, network):
        super().__init__()
        self.args = args
        self.model = network

        self.save_hyperparameters('args')

        self.datamodule = datamodule
        self.criterion = self.init_criterion()

        self.train_metrics = self.init_metrics()
        self.val_metrics = self.init_metrics()
        self.test_metrics = self.init_metrics()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> Dict[str, Any]:
        """Shared step logic for train/val/test"""
        x, y = batch

        # Forward pass
        logits = self(x)

        # Handle different output formats (some models return dict, others tensor)
        if isinstance(logits, dict):
            logits = logits['out']  # For some segmentation models

        # Ensure logits and targets have compatible shapes
        if logits.dim() == 4 and y.dim() == 3:
            # logits: (B, C, H, W), y: (B, H, W) - this is correct
            pass
        elif logits.dim() == 4 and y.dim() == 4:
            # logits: (B, C, H, W), y: (B, 1, H, W) - squeeze y
            y = y.squeeze(1)
        else:
            raise ValueError(f"Incompatible shapes: logits {logits.shape}, targets {y.shape}")

        # Calculate loss
        loss = self.criterion(logits, y)

        # Get predictions for metrics
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        metrics = getattr(self, f'{stage}_metrics')
        metrics.update(preds, y)

        # Log loss
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {
            'loss': loss,
            'preds': preds.detach(),
            'targets': y.detach()
        }

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        return self._shared_step(batch, 'val')

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        return self._shared_step(batch, 'test')

    def on_train_epoch_end(self) -> None:
        """Compute and log training metrics at epoch end"""
        self._log_epoch_metrics('train')

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at epoch end"""
        self._log_epoch_metrics('val')

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics at epoch end"""
        self._log_epoch_metrics('test')

    def _get_class_name(self, class_idx: int) -> str:
        """Get class name for logging"""
        if hasattr(self.datamodule, 'classes') and self.datamodule.classes:
            return self.datamodule.classes[class_idx]
        else:
            return f'class_{class_idx}'

    ########################
    # CRITERION & OPTIMIZER
    ########################

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        total_steps = self.args.epochs * len(self.datamodule.train_dataloader())
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.args.learning_rate,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step'  # Update every step, not epoch
            }
        }

    def init_criterion(self):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        return criterion

    def init_metrics(self):
        """Initialize metrics for segmentation"""
        metrics = MetricCollection({
            'accuracy': Accuracy(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='micro'
            ),
            'accuracy_macro': Accuracy(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='macro'
            ),
            'iou_micro': JaccardIndex(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='micro'
            ),
            'iou_macro': JaccardIndex(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='macro'
            ),
            'iou_per_class': JaccardIndex(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='none'
            ),
            'dice_micro': F1Score(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='micro'
            )
        })
        return metrics

    #################
    # LOGGING MODULE
    #################

    def _log_epoch_metrics(self, stage: str) -> None:
        """Helper to compute and log metrics at epoch end"""
        metrics = getattr(self, f'{stage}_metrics')
        computed_metrics = metrics.compute()

        for metric_name, metric_value in computed_metrics.items():
            if metric_value.dim() == 0:
                # Scalar metric
                self.log(f'{stage}_{metric_name}', metric_value, sync_dist=True)
            elif metric_value.dim() == 1:
                # Per-class metric
                if len(metric_value) == self.args.num_classes:
                    # Log mean
                    mean_val = metric_value.mean()
                    self.log(f'{stage}_{metric_name}_mean', mean_val, sync_dist=True)

                    # Log per-class values
                    for i, val in enumerate(metric_value):
                        class_name = self._get_class_name(i)
                        self.log(f'{stage}_{metric_name}_{class_name}', val, sync_dist=True)
                else:
                    # Just log the mean for other cases
                    self.log(f'{stage}_{metric_name}', metric_value.mean(), sync_dist=True)

        # Reset metrics for next epoch
        metrics.reset()


    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Prediction step for inference"""
        x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)

        logits = self(x)
        if isinstance(logits, dict):
            logits = logits['out']

        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_callbacks(self) -> List[Any]:
        """Configure additional callbacks if needed"""
        callbacks = []

        # Add gradient clipping if specified
        if hasattr(self.args, 'gradient_clip_val') and self.args.gradient_clip_val > 0:
            from lightning.pytorch.callbacks import GradientAccumulationScheduler
            # Note: Gradient clipping is typically handled in trainer config
            pass

        return callbacks

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        """Called before optimizer step - useful for gradient monitoring"""
        # Log gradient norms if needed
        if hasattr(self.args, 'log_gradients') and self.args.log_gradients:
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.log('gradient_norm', total_norm, on_step=True, on_epoch=False)
