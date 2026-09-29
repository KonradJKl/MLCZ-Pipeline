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

CLASS_NAMES = [
    'Background',
    'Compact high-rise', 'Compact mid-rise', 'Compact low-rise',
    'Open high-rise', 'Open mid-rise', 'Open low-rise',
    'Lightweight low-rise', 'Large low-rise', 'Sparsely built',
    'Heavy industry', 'Dense trees', 'Scattered trees',
    'Bush/scrub', 'Low plants', 'Bare rock/paved',
    'Bare soil/sand', 'Water'
]

# LCZ class colors (RGB values normalized to 0-1)
CLASS_COLORS = np.array([
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


class LCZVisualizer:
    """Plots and reports for LCZ predictions, saved to save_dir"""

    def __init__(self, save_dir: str = "./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = CLASS_NAMES
        self.class_colors = CLASS_COLORS

        # The first 10 channels are the S2 bands, in the order of BANDS in convert_data.py
        self.s2_band_indices = {
            'B02': 0, 'B03': 1, 'B04': 2, 'B05': 3, 'B06': 4, 'B07': 5, 'B08': 6, 'B8A': 7, 'B11': 8, 'B12': 9
        }

        # RGB band mapping for true color (B04=Red, B03=Green, B02=Blue)
        self.rgb_bands = [2, 1, 0]  # B04, B03, B02

        # False color composite (B08=NIR, B04=Red, B03=Green)
        self.false_color_bands = [6, 2, 1]  # B08, B04, B03

    def extract_s2_rgb(self, s2_data: Union[np.ndarray, torch.Tensor], use_false_color: bool = False) -> np.ndarray:
        """Return a true color (B04, B03, B02) or false color (B08, B04, B03) composite scaled to [0, 1]"""
        if isinstance(s2_data, torch.Tensor):
            s2_data = s2_data.cpu().numpy()

        bands = self.false_color_bands if use_false_color else self.rgb_bands

        if s2_data.ndim == 3:  # Single image (C, H, W)
            rgb = s2_data[bands].transpose(1, 2, 0)  # (H, W, 3)
        elif s2_data.ndim == 4:  # Batch (B, C, H, W)
            rgb = s2_data[:, bands].transpose(0, 2, 3, 1)  # (B, H, W, 3)
        else:
            raise ValueError(f"Unexpected S2 data shape: {s2_data.shape}")

        rgb = self.normalize_for_display(rgb)

        return rgb

    def normalize_for_display(self, data: np.ndarray, percentile_clip: Tuple[float, float] = (2, 98)) -> np.ndarray:
        """Clip data to the given (min, max) percentiles and scale it to [0, 1] for display"""
        p_min, p_max = np.percentile(data, percentile_clip)
        data_clipped = np.clip(data, p_min, p_max)

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
        """Compute the confusion matrix, save it as PNG and CSV and return it together with the figure"""
        try:
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()

            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

            if exclude_background:
                mask = y_true > 0
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                class_names = self.class_names[1:]  # Skip background
                labels = range(1, 18)
            else:
                class_names = self.class_names
                labels = range(18)

            if len(y_true) == 0:
                print("Warning: no valid samples for the confusion matrix")
                return None, None

            cm = confusion_matrix(y_true, y_pred, labels=labels)

            if normalize:
                cm_sum = cm.sum(axis=1)[:, np.newaxis]
                cm_sum[cm_sum == 0] = 1  # Avoid division by zero
                cm = cm.astype('float') / cm_sum
                cm = np.nan_to_num(cm)  # Handle any remaining NaN values

            fig, ax = plt.subplots(figsize=(16, 14))

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

            ax.set_ylabel('True Label', fontsize=14)
            ax.set_xlabel('Predicted Label', fontsize=14)
            ax.set_title(f'Confusion Matrix - {experiment_name}', fontsize=16)

            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

            plt.tight_layout()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.save_dir / f"confusion_matrix_{experiment_name}_{timestamp}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to: {save_path}")

            # Also save as CSV
            df = pd.DataFrame(cm, index=class_names, columns=class_names)
            csv_path = self.save_dir / f"confusion_matrix_{experiment_name}_{timestamp}.csv"
            df.to_csv(csv_path)
            print(f"Saved confusion matrix CSV to: {csv_path}")

            return cm, fig

        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def create_prediction_visualization(
            self,
            predictions: Union[np.ndarray, torch.Tensor],
            targets: Union[np.ndarray, torch.Tensor],
            s2_images: Union[np.ndarray, torch.Tensor] = None,
            experiment_name: str = "experiment",
            num_samples: int = 4,
            show_false_color: bool = True
    ) -> plt.Figure:
        """Plot S2 true and false color (if given), ground truth and prediction per sample and save the figure"""
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

        num_samples = min(num_samples, predictions.shape[0])

        if s2_images is not None:
            # 4 columns: S2 True Color, S2 False Color, Ground Truth, Prediction
            fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            col_titles = ['S2 True Color', 'S2 False Color', 'Ground Truth', 'Prediction']
        else:
            # 2 columns: Ground Truth, Prediction
            fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            col_titles = ['Ground Truth', 'Prediction']

        for i in range(num_samples):
            col_idx = 0

            if s2_images is not None:
                # The first 10 channels are the S2 bands
                s2_sample = s2_images[i, :10]

                s2_true_color = self.extract_s2_rgb(s2_sample, use_false_color=False)
                axes[i, col_idx].imshow(s2_true_color)
                axes[i, col_idx].set_title(f'S2 True Color - Sample {i + 1}')
                axes[i, col_idx].axis('off')
                col_idx += 1

                s2_false_color = self.extract_s2_rgb(s2_sample, use_false_color=True)
                axes[i, col_idx].imshow(s2_false_color)
                axes[i, col_idx].set_title(f'S2 False Color - Sample {i + 1}')
                axes[i, col_idx].axis('off')
                col_idx += 1

            target_colored = self.labels_to_colors(targets[i])
            pred_colored = self.labels_to_colors(predictions[i])

            axes[i, col_idx].imshow(target_colored)
            axes[i, col_idx].set_title(f'Ground Truth - Sample {i + 1}')
            axes[i, col_idx].axis('off')
            col_idx += 1

            axes[i, col_idx].imshow(pred_colored)
            axes[i, col_idx].set_title(f'Prediction - Sample {i + 1}')
            axes[i, col_idx].axis('off')

        plt.suptitle(f'{experiment_name} - S2 Data, Ground Truth & Predictions', fontsize=18)

        self.add_legend(fig)

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"predictions_with_s2_{experiment_name}_{timestamp}.png"
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved predictions visualization to: {save_path}")

        return fig

    def create_s2_band_visualization(
            self,
            s2_images: Union[np.ndarray, torch.Tensor],
            experiment_name: str = "experiment",
            num_samples: int = 2,
            bands_to_show: List[str] = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    ) -> plt.Figure:
        """Plot the S2 bands in bands_to_show in grayscale for the first num_samples samples and save the figure"""
        if isinstance(s2_images, torch.Tensor):
            s2_images = s2_images.cpu().numpy()

        num_samples = min(num_samples, s2_images.shape[0])
        num_bands = len(bands_to_show)

        fig, axes = plt.subplots(num_samples, num_bands, figsize=(4 * num_bands, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            for j, band_name in enumerate(bands_to_show):
                if band_name in self.s2_band_indices:
                    band_idx = self.s2_band_indices[band_name]
                    band_data = s2_images[i, band_idx]

                    band_normalized = self.normalize_for_display(band_data)

                    axes[i, j].imshow(band_normalized, cmap='gray')
                    axes[i, j].set_title(f'{band_name} - Sample {i + 1}')
                    axes[i, j].axis('off')
                else:
                    axes[i, j].text(0.5, 0.5, f'Band {band_name}\nnot found',
                                    ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')

        plt.suptitle(f'{experiment_name} - S2 Individual Bands', fontsize=18)
        plt.tight_layout()

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
        """Add the LCZ class legend to the figure"""
        # One entry per LCZ class, without background
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
        """Save the classification report without background as CSV and JSON and return the main metrics"""
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

        report = classification_report(
            y_true, y_pred,
            labels=range(1, 18),
            target_names=self.class_names[1:],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.save_dir / f"classification_report_{experiment_name}_{timestamp}.csv"
        report_df.to_csv(report_path)
        print(f"Saved classification report CSV to: {report_path}")

        comprehensive_metrics = {
            "overall_metrics": {
                # Not taken from the report, it has no "accuracy" entry as soon as a pixel is predicted as background
                "accuracy": float((y_true == y_pred).mean()),
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

        for i, class_name in enumerate(self.class_names[1:], 1):  # Skip background
            if class_name in report:
                comprehensive_metrics["per_class_metrics"][f"class_{i}_{class_name}"] = {
                    "precision": float(report[class_name]['precision']),
                    "recall": float(report[class_name]['recall']),
                    "f1-score": float(report[class_name]['f1-score']),
                    "support": int(report[class_name]['support'])
                }

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
            'comprehensive_metrics': comprehensive_metrics
        }

        return metrics


def visualize_model_predictions(model, dataloader, visualizer, experiment_name, device='cuda', max_batches=10):
    """Predict up to max_batches batches, save all plots and reports with the visualizer and return the metrics"""
    print(f"\nGenerating visualizations for {experiment_name}...")

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

            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']

            preds = torch.argmax(outputs, dim=1)

            # Store for metrics
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

            # Keep the first batch for the sample plots
            if i == 0:
                sample_preds = preds.cpu()
                sample_targets = labels.cpu()
                sample_images = images.cpu()

            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{max_batches} batches")

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    try:
        cm, cm_fig = visualizer.plot_confusion_matrix(
            all_targets,
            all_preds,
            experiment_name,
            normalize=True
        )

        if cm_fig is not None:
            # The figure stays open and is closed at the end, after the other plots
            print("Confusion matrix done")
        else:
            print("Could not generate the confusion matrix")

    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        cm, cm_fig = None, None

    try:
        pred_fig = visualizer.create_prediction_visualization(
            sample_preds,
            sample_targets,
            sample_images,
            experiment_name,
            num_samples=4
        )
        plt.close(pred_fig)  # Already saved
    except Exception as e:
        print(f"Error generating prediction visualization: {e}")

    try:
        band_fig = visualizer.create_s2_band_visualization(
            sample_images,
            experiment_name,
            num_samples=2,
            bands_to_show=['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
        )
        plt.close(band_fig)  # Already saved
    except Exception as e:
        print(f"Error generating S2 band visualization: {e}")

    metrics = visualizer.generate_metrics_report(
        all_targets,
        all_preds,
        experiment_name
    )

    print("\nMetrics summary:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Macro F1: {metrics['macro_f1']:.3f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.3f}")

    # Close the confusion matrix figure only after everything else is done
    if cm_fig is not None:
        # Give the figure a moment to be drawn
        plt.pause(0.1)
        plt.close(cm_fig)

    return metrics