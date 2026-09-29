"""Stitch the patch predictions of a checkpoint into city maps and score them, used by app.py and the page export"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from modules.MLCZ import MLCZIndexableLMDBDataset
from modules.model import get_network
from modules.visualization import CLASS_NAMES, CLASS_COLORS

PATCH = 64
NUM_CLASSES = len(CLASS_NAMES)
# B4, B3, B2: channels 0-9 are the S2 bands B2 to B12, followed by the 234 PRISMA bands
TRUE_COLOR = [2, 1, 0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLORS = (CLASS_COLORS * 255).round().astype(np.uint8)
ERROR_COLORS = np.array([[0, 0, 0], [200, 200, 200], [220, 20, 60]], dtype=np.uint8)  # No label, correct, wrong
LCZ_CODES = [""] + [str(i) for i in range(1, 11)] + list("ABCDEFG")
LABELS = ["No data"] + [f"LCZ {code} {name}" for code, name in zip(LCZ_CODES[1:], CLASS_NAMES[1:])]


def find_checkpoints(logs_dir: Path):
    """All checkpoints below logs_dir, newest first"""
    return sorted(Path(logs_dir).rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_model(ckpt_path):
    """Rebuild the network of a checkpoint from experiments.py, returns it in eval mode on DEVICE and its args"""
    # weights_only=False because the hyperparameters hold a Namespace, so only load own checkpoints
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = checkpoint["hyper_parameters"]["args"]
    model = get_network(args.arch_name, args.num_channels, args.num_classes, pretrained=False, dropout=args.dropout)
    # Lightning stores the network under "model.", other keys like the loss weights are not needed
    weights = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(weights)
    return model.eval().to(DEVICE), args


def predict_city(model, lmdb_path: Path, parquet_path: Path, city: str, batch_size: int = 32):
    """
    Predict all patches of a city and stitch them into maps in raster coordinates.

    :return: dict with "rgb" (H, W, 3), "label" and "pred" (H, W) as uint8 and "confusion", one matrix per split
    """
    start = time.time()
    dataset = MLCZIndexableLMDBDataset(str(lmdb_path), str(parquet_path), split=None, cities=[city])
    meta = dataset.metadata
    rows, cols, splits = meta["row_offset"].to_numpy(), meta["col_offset"].to_numpy(), meta["split"].to_numpy()
    height, width = rows.max() + PATCH, cols.max() + PATCH

    probs = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
    bands = np.zeros((3, height, width), dtype=np.float32)
    label = np.zeros((height, width), dtype=np.uint8)
    covered = np.zeros((height, width), dtype=bool)
    confusions = {split: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for split in np.unique(splits)}

    # No workers (on Windows each one imports the calling script again) and no shuffling, so patch i is row i of meta
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    i = 0
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            batch_probs = logits.softmax(1).cpu().numpy()
            batch_preds = logits.argmax(1).cpu().numpy()
            for patch_probs, patch_pred, image, target in zip(batch_probs, batch_preds, x.numpy(), y.numpy()):
                r, c = rows[i], cols[i]
                probs[:, r:r + PATCH, c:c + PATCH] += patch_probs
                # Overlapping windows are cut from the same rasters, so overwriting is enough
                bands[:, r:r + PATCH, c:c + PATCH] = image[TRUE_COLOR]
                label[r:r + PATCH, c:c + PATCH] = target
                covered[r:r + PATCH, c:c + PATCH] = True
                # Metrics use each patch's own prediction, like trainer.test
                confusions[splits[i]] += confusion(target, patch_pred)
                i += 1

    # The sum has the same argmax as the mean, uncovered pixels are all zero and become class 0
    pred = probs.argmax(0).astype(np.uint8)
    print(f"{city}: predicted {len(dataset)} patches in {time.time() - start:.1f} s on {DEVICE}")
    return {"rgb": stretch(bands, covered), "label": label, "pred": pred, "confusion": confusions}


def stretch(bands: np.ndarray, valid: np.ndarray):
    """(3, H, W) bands to an (H, W, 3) uint8 image, one 2-98 percentile stretch over the valid pixels, others black"""
    low, high = np.percentile(bands[:, valid], (2, 98))
    rgb = np.zeros((*valid.shape, 3), dtype=np.uint8)
    rgb[valid] = (np.clip((bands[:, valid].T - low) / (high - low), 0, 1) * 255).round().astype(np.uint8)
    return rgb


def confusion(labels, preds):
    """Confusion matrix with the true class in the rows and the predicted class in the columns"""
    index = NUM_CLASSES * labels.astype(np.int64).ravel() + preds.ravel()
    return np.bincount(index, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def class_table(cm: np.ndarray):
    """Scores of every class found in the labels or predictions, the mean of the IoU column is the mean IoU"""
    # Label 0 is ignored, but predicting 0 on a labelled pixel still counts as wrong
    tp = cm.diagonal()[1:]
    pixels = cm[1:].sum(1)
    predicted = cm[1:, 1:].sum(0)
    union = pixels + predicted - tp
    # tp is 0 wherever a count is 0, so max(count, 1) turns 0 / 0 into 0
    table = pd.DataFrame({
        "LCZ": LABELS[1:],
        "Pixels": pixels,
        "Precision": tp / np.maximum(predicted, 1),
        "Recall": tp / np.maximum(pixels, 1),
        "F1": 2 * tp / np.maximum(pixels + predicted, 1),
        "IoU": tp / np.maximum(union, 1),
    })
    return table[union > 0].reset_index(drop=True)


def split_table(confusions: dict):
    """Accuracy and mean IoU per split, label 0 is ignored like in the test metrics of base.py"""
    rows = {}
    for split in ["train", "validation", "test"]:
        if split in confusions:
            cm = confusions[split]
            rows[split] = {"Labelled pixels": cm[1:].sum(),
                           "Accuracy": cm[1:, 1:].trace() / cm[1:].sum(),
                           "Mean IoU": class_table(cm)["IoU"].mean()}
    return pd.DataFrame.from_dict(rows, orient="index")


def panels(result: dict, window=np.s_[:, :]):
    """The four maps of a predict_city result as (h, w, 3) uint8 images, cropped to window"""
    label, pred = result["label"][window], result["pred"][window]
    errors = np.where(label == 0, 0, np.where(pred == label, 1, 2))
    return {"Sentinel-2 true color": result["rgb"][window], "Ground truth": COLORS[label],
            "Prediction": COLORS[pred], "Errors": ERROR_COLORS[errors]}


def legend_html(classes):
    """Color chips with the LCZ names, inline styles so it works in Streamlit and on the static page"""
    chips = []
    for k in classes:
        r, g, b = COLORS[k]
        chips.append('<span style="display:inline-flex;align-items:center;gap:6px;margin:0 16px 6px 0">'
                     f'<span style="width:14px;height:14px;border:1px solid #888;background:rgb({r},{g},{b})"></span>'
                     f"{LABELS[k]}</span>")
    return f'<div style="display:flex;flex-wrap:wrap">{"".join(chips)}</div>'
