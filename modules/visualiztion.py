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


class LCZVisualizer:
    """Visualization tools for LCZ classification that work with existing code."""

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

        Args:
            y_true: True labels (can be 2D/3D array or flattened)
            y_pred: Predicted labels (can be 2D/3D array or flattened)
            experiment_name: Name for saving the plot
            normalize: Whether to normalize the confusion matrix
            exclude_background: Whether to exclude background class (0)

        Returns:
            Confusion matrix array and matplotlib figure
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
            experiment_name: str = "experiment",
            num_samples: int = 4
    ) -> plt.Figure:
        """
        Create visualization comparing predictions with ground truth.

        Args:
            predictions: Predicted labels (B, H, W)
            targets: True labels (B, H, W)
            experiment_name: Name for saving
            num_samples: Number of samples to visualize

        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        # Ensure we have batch dimension
        if predictions.ndim == 2:
            predictions = predictions[np.newaxis, ...]
            targets = targets[np.newaxis, ...]

        # Limit number of samples
        num_samples = min(num_samples, predictions.shape[0])

        # Create figure
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Convert labels to colors
            pred_colored = self.labels_to_colors(predictions[i])
            target_colored = self.labels_to_colors(targets[i])

            # Plot ground truth
            axes[i, 0].imshow(target_colored)
            axes[i, 0].set_title(f'Ground Truth - Sample {i + 1}')
            axes[i, 0].axis('off')

            # Plot prediction
            axes[i, 1].imshow(pred_colored)
            axes[i, 1].set_title(f'Prediction - Sample {i + 1}')
            axes[i, 1].axis('off')

        plt.suptitle(f'{experiment_name} - Predictions vs Ground Truth', fontsize=16)

        # Add legend
        self.add_legend(fig)

        plt.tight_layout()

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"predictions_{experiment_name}_{timestamp}.png"
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved predictions visualization to: {save_path}")

        return fig

    def labels_to_colors(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert label map to RGB colors.

        Args:
            labels: Label array (H, W)

        Returns:
            RGB array (H, W, 3)
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
        Generate classification metrics report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            experiment_name: Name for saving

        Returns:
            Dictionary of metrics
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

        # Save detailed report
        report_df = pd.DataFrame(report).transpose()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.save_dir / f"classification_report_{experiment_name}_{timestamp}.csv"
        report_df.to_csv(report_path)
        print(f"Saved classification report to: {report_path}")

        # Extract key metrics
        metrics = {
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall']
        }

        return metrics


def visualize_model_predictions(model, dataloader, visualizer, experiment_name, device='cuda', max_batches=10):
    """
    Helper function to visualize model predictions on test data.

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

            # Store first batch for visualization
            if i == 0:
                sample_preds = preds.cpu()
                sample_targets = labels.cpu()

    # Concatenate all predictions
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Confusion Matrix
    cm, cm_fig = visualizer.plot_confusion_matrix(
        all_targets,
        all_preds,
        experiment_name,
        normalize=True
    )
    plt.close(cm_fig)

    # 2. Sample predictions
    pred_fig = visualizer.create_prediction_visualization(
        sample_preds,
        sample_targets,
        experiment_name,
        num_samples=4
    )
    plt.close(pred_fig)

    # 3. Metrics report
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


# Example usage that can be added to experiments.py after training:
"""
# After trainer.test(model, ckpt_path="best") in experiments.py, add:

from visualization import LCZVisualizer, visualize_model_predictions

# Create visualizer
visualizer = LCZVisualizer(save_dir="./visualizations")

# Generate visualizations on test set
experiment_name = f"{args.dataset}_{args.arch_name}_pt={args.pretrained}_do={args.dropout}"
metrics = visualize_model_predictions(
    model,
    datamodule.test_dataloader(),
    visualizer,
    experiment_name,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
"""