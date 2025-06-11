import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.patches import Patch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json


class LCZVisualizer:
    """Enhanced visualization tools for LCZ classification with S2 data display."""

    def __init__(self, save_dir: str = "./visualizations"):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # LCZ class names
        self.class_names = [
            'Background',
            'Compact high-rise', 'Compact mid-rise', 'Compact low-rise',
            'Open high-rise', 'Open mid-rise', 'Open low-rise',
            'Lightweight low-rise', 'Large low-rise', 'Sparsely built',
            'Heavy industry', 'Dense trees', 'Scattered trees',
            'Bush/scrub', 'Low plants', 'Bare rock/paved',
            'Bare soil/sand', 'Water'
        ]

        # LCZ class colors (RGB values normalized to 0-1)
        self.class_colors = np.array([
            [0, 0, 0],  # Background - Black
            [139, 0, 0],  # Compact high-rise - Dark Red
            [255, 0, 0],  # Compact mid-rise - Red
            [255, 165, 0],  # Compact low-rise - Orange
            [255, 255, 0],  # Open high-rise - Yellow
            [255, 255, 128],  # Open mid-rise - Light Yellow
            [255, 255, 192],  # Open low-rise - Very Light Yellow
            [192, 192, 192],  # Lightweight low-rise - Light Gray
            [128, 128, 128],  # Large low-rise - Gray
            [255, 192, 203],  # Sparsely built - Pink
            [128, 0, 128],  # Heavy industry - Purple
            [0, 128, 0],  # Dense trees - Dark Green
            [0, 255, 0],  # Scattered trees - Green
            [128, 255, 128],  # Bush/scrub - Light Green
            [0, 255, 255],  # Low plants - Cyan
            [128, 128, 0],  # Bare rock/paved - Olive
            [255, 228, 196],  # Bare soil/sand - Bisque
            [0, 0, 255]  # Water - Blue
        ]) / 255.0

        # S2 band indices (assuming S2 bands come first in the tensor)
        # Based on your BANDS list: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
        self.s2_band_indices = {
            'B01': 0, 'B02': 1, 'B03': 2, 'B04': 3, 'B05': 4, 'B06': 5, 'B07': 6,
            'B08': 7, 'B09': 8, 'B10': 9, 'B11': 10, 'B12': 11, 'B8A': 12
        }

        # RGB band mapping for true color (B04=Red, B03=Green, B02=Blue)
        self.rgb_bands = [3, 2, 1]  # B04, B03, B02

        # False color composite (B08=NIR, B04=Red, B03=Green)
        self.false_color_bands = [7, 3, 2]  # B08, B04, B03

    def extract_s2_rgb(self, s2_data: Union[np.ndarray, torch.Tensor], use_false_color: bool = False) -> np.ndarray:
        """
        Extract RGB visualization from S2 data.

        Args:
            s2_data: S2 tensor of shape (C, H, W) or (B, C, H, W)
            use_false_color: If True, use false color composite (NIR, Red, Green)
                            If False, use true color (Red, Green, Blue)

        Returns:
            RGB array of shape (H, W, 3) or (B, H, W, 3)
        """
        if isinstance(s2_data, torch.Tensor):
            s2_data = s2_data.cpu().numpy()

        # Choose bands
        bands = self.false_color_bands if use_false_color else self.rgb_bands

        if s2_data.ndim == 3:  # Single image (C, H, W)
            rgb = s2_data[bands].transpose(1, 2, 0)  # (H, W, 3)
        elif s2_data.ndim == 4:  # Batch (B, C, H, W)
            rgb = s2_data[:, bands].transpose(0, 2, 3, 1)  # (B, H, W, 3)
        else:
            raise ValueError(f"Unexpected S2 data shape: {s2_data.shape}")

        # Normalize to 0-1 range for visualization
        rgb = self.normalize_for_display(rgb)

        return rgb

    def normalize_for_display(self, data: np.ndarray, percentile_clip: Tuple[float, float] = (2, 98)) -> np.ndarray:
        """
        Normalize data for display using percentile clipping.

        Args:
            data: Input data array
            percentile_clip: (min_percentile, max_percentile) for clipping

        Returns:
            Normalized data in range [0, 1]
        """
        # Clip outliers using percentiles
        p_min, p_max = np.percentile(data, percentile_clip)
        data_clipped = np.clip(data, p_min, p_max)

        # Normalize to 0-1
        data_norm = (data_clipped - p_min) / (p_max - p_min + 1e-8)

        return np.clip(data_norm, 0, 1)

    def plot_confusion_matrix(
            self,
            y_true: Union[np.ndarray, torch.Tensor],
            y_pred: Union[np.ndarray, torch.Tensor],
            experiment_name: str = "experiment",
            normalize: bool = True,
            exclude_background: bool = True
    ) -> Tuple[np.ndarray, plt.Figure]:
        """
        Generate and save confusion matrix.
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        # Flatten arrays
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Optionally exclude background
        if exclude_background:
            mask = y_true > 0
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            class_names = self.class_names[1:]  # Skip background
            labels = range(1, 18)
        else:
            class_names = self.class_names
            labels = range(18)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # Handle division by zero

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'},
            annot_kws={'size': 8}
        )

        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {experiment_name}', fontsize=14)

        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"confusion_matrix_{experiment_name}_{timestamp}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to: {save_path}")

        # Also save as CSV
        df = pd.DataFrame(cm, index=class_names, columns=class_names)
        csv_path = self.save_dir / f"confusion_matrix_{experiment_name}_{timestamp}.csv"
        df.to_csv(csv_path)

        return cm, fig

    def create_prediction_visualization(
            self,
            predictions: Union[np.ndarray, torch.Tensor],
            targets: Union[np.ndarray, torch.Tensor],
            s2_images: Union[np.ndarray, torch.Tensor] = None,
            experiment_name: str = "experiment",
            num_samples: int = 4,
            show_false_color: bool = True
    ) -> plt.Figure:
        """
        Create enhanced visualization with S2 data, predictions and ground truth.

        Args:
            predictions: Predicted labels (B, H, W)
            targets: True labels (B, H, W)
            s2_images: S2 satellite images (B, C, H, W) - first 13 channels should be S2
            experiment_name: Name for saving
            num_samples: Number of samples to visualize
            show_false_color: If True, show false color composite; if False, show true color

        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if s2_images is not None and isinstance(s2_images, torch.Tensor):
            s2_images = s2_images.cpu().numpy()

        # Ensure we have batch dimension
        if predictions.ndim == 2:
            predictions = predictions[np.newaxis, ...]
            targets = targets[np.newaxis, ...]
            if s2_images is not None:
                s2_images = s2_images[np.newaxis, ...]

        # Limit number of samples
        num_samples = min(num_samples, predictions.shape[0])

        # Determine subplot layout
        if s2_images is not None:
            # 4 columns: S2 True Color, S2 False Color, Ground Truth, Prediction
            fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            col_titles = ['S2 True Color', 'S2 False Color', 'Ground Truth', 'Prediction']
        else:
            # 2 columns: Ground Truth, Prediction (original layout)
            fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            col_titles = ['Ground Truth', 'Prediction']

        for i in range(num_samples):
            col_idx = 0

            # Show S2 data if available
            if s2_images is not None:
                # Extract S2 bands (assuming first 13 channels are S2)
                s2_sample = s2_images[i, :13]  # Take first 13 channels (S2 bands)

                # True color RGB
                s2_true_color = self.extract_s2_rgb(s2_sample, use_false_color=False)
                axes[i, col_idx].imshow(s2_true_color)
                axes[i, col_idx].set_title(f'S2 True Color - Sample {i + 1}')
                axes[i, col_idx].axis('off')
                col_idx += 1

                # False color composite
                s2_false_color = self.extract_s2_rgb(s2_sample, use_false_color=True)
                axes[i, col_idx].imshow(s2_false_color)
                axes[i, col_idx].set_title(f'S2 False Color - Sample {i + 1}')
                axes[i, col_idx].axis('off')
                col_idx += 1

            # Convert labels to colors
            target_colored = self.labels_to_colors(targets[i])
            pred_colored = self.labels_to_colors(predictions[i])

            # Plot ground truth
            axes[i, col_idx].imshow(target_colored)
            axes[i, col_idx].set_title(f'Ground Truth - Sample {i + 1}')
            axes[i, col_idx].axis('off')
            col_idx += 1

            # Plot prediction
            axes[i, col_idx].imshow(pred_colored)
            axes[i, col_idx].set_title(f'Prediction - Sample {i + 1}')
            axes[i, col_idx].axis('off')

        plt.suptitle(f'{experiment_name} - S2 Data, Ground Truth & Predictions', fontsize=16)

        # Add legend for LCZ classes
        self.add_legend(fig)

        plt.tight_layout()

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"predictions_with_s2_{experiment_name}_{timestamp}.png"
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved enhanced predictions visualization to: {save_path}")

        return fig

    def create_s2_band_visualization(
            self,
            s2_images: Union[np.ndarray, torch.Tensor],
            experiment_name: str = "experiment",
            num_samples: int = 2,
            bands_to_show: List[str] = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    ) -> plt.Figure:
        """
        Create visualization showing individual S2 bands.

        Args:
            s2_images: S2 satellite images (B, C, H, W)
            experiment_name: Name for saving
            num_samples: Number of samples to visualize
            bands_to_show: List of S2 band names to display

        Returns:
            Matplotlib figure
        """
        if isinstance(s2_images, torch.Tensor):
            s2_images = s2_images.cpu().numpy()

        num_samples = min(num_samples, s2_images.shape[0])
        num_bands = len(bands_to_show)

        fig, axes = plt.subplots(num_samples, num_bands, figsize=(3 * num_bands, 3 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            for j, band_name in enumerate(bands_to_show):
                if band_name in self.s2_band_indices:
                    band_idx = self.s2_band_indices[band_name]
                    band_data = s2_images[i, band_idx]

                    # Normalize for display
                    band_normalized = self.normalize_for_display(band_data)

                    axes[i, j].imshow(band_data, cmap='gray')
                    axes[i, j].set_title(f'{band_name} - Sample {i + 1}')
                    axes[i, j].axis('off')
                else:
                    axes[i, j].text(0.5, 0.5, f'Band {band_name}\nnot found',
                                    ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')

        plt.suptitle(f'{experiment_name} - S2 Individual Bands', fontsize=16)
        plt.tight_layout()

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"s2_bands_{experiment_name}_{timestamp}.png"
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved S2 bands visualization to: {save_path}")

        return fig

    def labels_to_colors(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert label map to RGB colors.
        """
        H, W = labels.shape
        rgb = np.zeros((H, W, 3))

        for class_idx in range(len(self.class_colors)):
            mask = labels == class_idx
            rgb[mask] = self.class_colors[class_idx]

        return rgb

    def add_legend(self, fig: plt.Figure):
        """Add legend to figure."""
        # Create legend elements (skip background)
        legend_elements = [
            Patch(facecolor=self.class_colors[i], label=self.class_names[i])
            for i in range(1, 18)
        ]

        # Add legend to the right of the figure
        fig.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            ncol=1,
            fontsize=10
        )

    def generate_metrics_report(
            self,
            y_true: Union[np.ndarray, torch.Tensor],
            y_pred: Union[np.ndarray, torch.Tensor],
            experiment_name: str = "experiment"
    ) -> Dict[str, float]:
        """
        Generate comprehensive classification metrics report and save as JSON.

        Returns:
            Dict containing both overall metrics and per-class metrics
        """
        # Convert to numpy and flatten
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Remove background
        mask = y_true > 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Generate report
        report = classification_report(
            y_true, y_pred,
            labels=range(1, 18),
            target_names=self.class_names[1:],
            output_dict=True,
            zero_division=0
        )

        # Save detailed report as CSV
        report_df = pd.DataFrame(report).transpose()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.save_dir / f"classification_report_{experiment_name}_{timestamp}.csv"
        report_df.to_csv(report_path)
        print(f"Saved classification report CSV to: {report_path}")

        # Create comprehensive metrics dictionary
        comprehensive_metrics = {
            "overall_metrics": {
                "accuracy": float(report['accuracy']),
                "macro_avg": {
                    "precision": float(report['macro avg']['precision']),
                    "recall": float(report['macro avg']['recall']),
                    "f1-score": float(report['macro avg']['f1-score']),
                    "support": int(report['macro avg']['support'])
                },
                "weighted_avg": {
                    "precision": float(report['weighted avg']['precision']),
                    "recall": float(report['weighted avg']['recall']),
                    "f1-score": float(report['weighted avg']['f1-score']),
                    "support": int(report['weighted avg']['support'])
                }
            },
            "per_class_metrics": {}
        }

        # Add per-class metrics
        for i, class_name in enumerate(self.class_names[1:], 1):  # Skip background
            if class_name in report:
                comprehensive_metrics["per_class_metrics"][f"class_{i}_{class_name}"] = {
                    "precision": float(report[class_name]['precision']),
                    "recall": float(report[class_name]['recall']),
                    "f1-score": float(report[class_name]['f1-score']),
                    "support": int(report[class_name]['support'])
                }

        # Save as JSON
        json_path = self.save_dir / f"metrics_{experiment_name}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(comprehensive_metrics, f, indent=4)
        print(f"Saved metrics JSON to: {json_path}")

        # Return both simple metrics (for backward compatibility) and full metrics
        metrics = {
            'accuracy': comprehensive_metrics['overall_metrics']['accuracy'],
            'macro_f1': comprehensive_metrics['overall_metrics']['macro_avg']['f1-score'],
            'weighted_f1': comprehensive_metrics['overall_metrics']['weighted_avg']['f1-score'],
            'macro_precision': comprehensive_metrics['overall_metrics']['macro_avg']['precision'],
            'macro_recall': comprehensive_metrics['overall_metrics']['macro_avg']['recall'],
            'comprehensive_metrics': comprehensive_metrics  # Add full metrics
        }

        return metrics


def visualize_model_predictions(model, dataloader, visualizer, experiment_name, device='cuda', max_batches=10):
    """
    Enhanced helper function to visualize model predictions with S2 data.

    Args:
        model: Trained model
        dataloader: Test dataloader
        visualizer: LCZVisualizer instance
        experiment_name: Name of experiment
        device: Device to use
        max_batches: Maximum number of batches to process
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    sample_preds = []
    sample_targets = []
    sample_images = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            # Get predictions
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']

            preds = torch.argmax(outputs, dim=1)

            # Store for metrics
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

            # Store first batch for visualization (including images)
            if i == 0:
                sample_preds = preds.cpu()
                sample_targets = labels.cpu()
                sample_images = images.cpu()

    # Concatenate all predictions
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Generate visualizations
    print("\nGenerating enhanced visualizations with S2 data...")

    # 1. Confusion Matrix
    cm, cm_fig = visualizer.plot_confusion_matrix(
        all_targets,
        all_preds,
        experiment_name,
        normalize=True
    )
    plt.close(cm_fig)

    # 2. Enhanced predictions with S2 data
    pred_fig = visualizer.create_prediction_visualization(
        sample_preds,
        sample_targets,
        sample_images,  # Pass the S2 images
        experiment_name,
        num_samples=4
    )
    plt.close(pred_fig)

    # 3. S2 band visualization
    band_fig = visualizer.create_s2_band_visualization(
        sample_images,
        experiment_name,
        num_samples=2,
        bands_to_show=['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    )
    plt.close(band_fig)

    # 4. Metrics report (now includes JSON output)
    metrics = visualizer.generate_metrics_report(
        all_targets,
        all_preds,
        experiment_name
    )

    print(f"\nMetrics Summary:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Macro F1: {metrics['macro_f1']:.3f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.3f}")

    return metrics