import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torchmetrics import MetricCollection, JaccardIndex, Accuracy, F1Score
from torchmetrics.segmentation import DiceScore
import torch.nn.functional as F
import numpy as np

from typing import Any, Dict, List, Tuple


class CombinedLoss(nn.Module):
    """Weighted sum of cross-entropy and focal loss"""

    def __init__(self, num_classes, ignore_index=0, alpha=0.7, gamma=2.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha  # Weight between CE and Focal loss
        self.gamma = gamma  # Focal loss gamma parameter

        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=class_weights
        )

    def focal_loss(self, inputs, targets):
        """Focal loss to handle class imbalance"""
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)

        return self.alpha * ce + (1 - self.alpha) * focal


class BaseModel(L.LightningModule):
    def __init__(self, args, datamodule, network):
        super().__init__()
        self.args = args
        self.model = network

        self.save_hyperparameters('args')

        self.datamodule = datamodule

        self.criterion = self.init_enhanced_criterion()

        self.train_metrics = self.init_enhanced_metrics()
        self.val_metrics = self.init_enhanced_metrics()
        self.test_metrics = self.init_enhanced_metrics()

        self.automatic_optimization = True

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> Dict[str, Any]:
        """Shared step for train, val and test"""
        x, y = batch

        logits = self(x)

        # Some segmentation models return a dict instead of a tensor
        if isinstance(logits, dict):
            logits = logits['out']

        # Make logits and targets shapes compatible
        if logits.dim() == 4 and y.dim() == 3:
            # logits (B, C, H, W), y (B, H, W)
            pass
        elif logits.dim() == 4 and y.dim() == 4:
            # logits (B, C, H, W), y (B, 1, H, W)
            y = y.squeeze(1)
        else:
            raise ValueError(f"Incompatible shapes: logits {logits.shape}, targets {y.shape}")

        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        metrics = getattr(self, f'{stage}_metrics')
        metrics.update(preds, y)

        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if stage == 'train':
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=True, on_epoch=False)

        return {
            'loss': loss,
            'preds': preds.detach(),
            'targets': y.detach(),
            'logits': logits.detach()
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
        # LCZ class names used in the per-class metric keys
        lcz_names = {
            0: 'background',
            1: 'compact_high_rise', 2: 'compact_mid_rise', 3: 'compact_low_rise',
            4: 'open_high_rise', 5: 'open_mid_rise', 6: 'open_low_rise',
            7: 'lightweight_low_rise', 8: 'large_low_rise', 9: 'sparsely_built',
            10: 'heavy_industry', 11: 'dense_trees', 12: 'scattered_trees',
            13: 'bush_scrub', 14: 'low_plants', 15: 'bare_rock_paved',
            16: 'bare_soil_sand', 17: 'water'
        }

        return lcz_names.get(class_idx, f'class_{class_idx}')

    ########################
    # CRITERION & OPTIMIZER
    ########################

    def configure_optimizers(self):
        """AdamW with a learning rate scheduler chosen by the number of epochs"""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        total_steps = self.args.epochs * len(self.datamodule.train_dataloader())

        if self.args.epochs <= 50:
            # OneCycle for shorter training
            lr_scheduler = OneCycleLR(
                optimizer,
                max_lr=self.args.learning_rate,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25,
                final_div_factor=1000
            )
            scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step'
            }
        else:
            # Cosine annealing for longer training
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.args.epochs,
                eta_min=self.args.learning_rate * 0.01
            )
            scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'epoch'
            }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }

    def calculate_class_weights(self):
        """Inverse frequency class weights from up to 100 training batches"""
        print("Calculating class weights from training data...")

        class_counts = torch.zeros(self.args.num_classes)
        total_pixels = 0

        train_loader = self.datamodule.train_dataloader()
        sample_size = min(100, len(train_loader))

        for i, (_, targets) in enumerate(train_loader):
            if i >= sample_size:
                break

            for class_idx in range(self.args.num_classes):
                class_counts[class_idx] += (targets == class_idx).sum().item()
            total_pixels += targets.numel()

        class_weights = total_pixels / (self.args.num_classes * class_counts)
        class_weights[class_counts == 0] = 0  # Classes without pixels get weight 0

        # Normalize so the weights sum to num_classes
        class_weights = class_weights / class_weights.sum() * self.args.num_classes

        print(f"  Class weights: {class_weights}")
        return class_weights

    def init_enhanced_criterion(self):
        """Combined CE and focal loss, optionally with class weights"""
        class_weights = None

        if hasattr(self.args, 'class_weights') and self.args.class_weights:
            try:
                class_weights = self.calculate_class_weights()
                class_weights = class_weights.to(self.device)
            except Exception as e:
                print(f"Could not calculate class weights: {e}")
                class_weights = None

        # Focal loss part helps with the imbalanced LCZ classes
        criterion = CombinedLoss(
            num_classes=self.args.num_classes,
            ignore_index=0,
            alpha=0.7,  # Balance between CE and focal loss
            gamma=2.0,
            class_weights=class_weights
        )

        return criterion

    def init_enhanced_metrics(self):
        """Accuracy, IoU and F1 metrics, ignoring the background class 0"""
        metrics = MetricCollection({
            # Accuracy metrics
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
            'accuracy_weighted': Accuracy(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='weighted'
            ),

            # IoU metrics
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
            'iou_weighted': JaccardIndex(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='weighted'
            ),
            'iou_per_class': JaccardIndex(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='none'
            ),

            # F1 scores
            'f1_micro': F1Score(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='micro'
            ),
            'f1_macro': F1Score(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='macro'
            ),
            'f1_weighted': F1Score(
                task='multiclass',
                num_classes=self.args.num_classes,
                ignore_index=0,
                average='weighted'
            ),
        })
        return metrics

    #################
    # LOGGING MODULE
    #################

    def _log_epoch_metrics(self, stage: str) -> None:
        """Compute, log and reset the metrics of one stage"""
        metrics = getattr(self, f'{stage}_metrics')
        computed_metrics = metrics.compute()

        for metric_name, metric_value in computed_metrics.items():
            if metric_value.dim() == 0:
                # Scalar metric
                self.log(f'{stage}_{metric_name}', metric_value, sync_dist=True)
            elif metric_value.dim() == 1:
                # Per-class metric, log the mean and one value per class
                if len(metric_value) == self.args.num_classes:
                    mean_val = metric_value.mean()
                    self.log(f'{stage}_{metric_name}_mean', mean_val, sync_dist=True)

                    for i, val in enumerate(metric_value):
                        class_name = self._get_class_name(i)
                        self.log(f'{stage}_{metric_name}_{class_name}', val, sync_dist=True)
                else:
                    self.log(f'{stage}_{metric_name}', metric_value.mean(), sync_dist=True)

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

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        """Log the total gradient norm if args.log_gradients is set"""
        if hasattr(self.args, 'log_gradients') and self.args.log_gradients:
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.log('gradient_norm', total_norm, on_step=True, on_epoch=False)
