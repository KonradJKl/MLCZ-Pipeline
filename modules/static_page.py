"""Write the city maps and scores of one checkpoint for the project page in docs/, used by main.py --export"""
import json
from datetime import date
from pathlib import Path

import pandas as pd
from PIL import Image

from modules.city_maps import COLORS, LABELS, LCZ_CODES, load_model, predict_city, split_table, class_table, panels
from modules.visualization import CLASS_NAMES

ARCHS = {"unet": "U-Net with a ResNet18 encoder", "CustomCNN": "Small U-Net style CNN"}
# Keys in results.json and the titles of city_maps.panels
IMAGES = {"s2": "Sentinel-2 true color", "label": "Ground truth", "pred": "Prediction", "errors": "Errors"}
OVERVIEW_CITY = "Berlin"
OVERVIEW_WIDTH = 1600
GAP = 16


def rounded(value):
    return round(float(value), 4)


def save_maps(slug: str, result: dict, out_dir: Path):
    """Save the four maps of a city into out_dir/maps and return their paths relative to out_dir"""
    images = panels(result)
    paths = {}
    for key, title in IMAGES.items():
        # Lowercase names, GitHub Pages is case-sensitive. PNG keeps the class colors exact, the page reads them back
        suffix = "jpg" if key == "s2" else "png"
        paths[key] = f"maps/{slug}_{key}.{suffix}"
        # PNG ignores quality
        Image.fromarray(images[title]).save(out_dir / paths[key], quality=80, optimize=True)
    return paths


def save_overview(result: dict, path: Path):
    """Sentinel-2 true color, ground truth and prediction side by side, the image at the top of the README"""
    images = panels(result)
    height, width = result["label"].shape
    scale = min(1, (OVERVIEW_WIDTH - 2 * GAP) / (3 * width))
    size = (int(width * scale), int(height * scale))
    # Transparent gaps, so the image fits light and dark themes
    overview = Image.new("RGBA", (3 * size[0] + 2 * GAP, size[1]))
    for i, title in enumerate([IMAGES["s2"], IMAGES["label"], IMAGES["pred"]]):
        # Nearest neighbor keeps the class colors of the maps
        resample = Image.Resampling.LANCZOS if i == 0 else Image.Resampling.NEAREST
        overview.paste(Image.fromarray(images[title]).resize(size, resample), (i * (size[0] + GAP), 0))
    overview.save(path, optimize=True)


def scores(confusions: dict):
    """Pixels, accuracy and mean IoU per split and the scores per class on the test split"""
    splits = {split: {"pixels": int(row["Labelled pixels"]), "accuracy": rounded(row["Accuracy"]),
                      "mean_iou": rounded(row["Mean IoU"])} for split, row in split_table(confusions).iterrows()}
    # A city_based split can leave a city without test patches
    table = class_table(confusions["test"]) if "test" in confusions else pd.DataFrame()
    test_classes = [{"id": LABELS.index(row.LCZ), "pixels": int(row.Pixels), "precision": rounded(row.Precision),
                     "recall": rounded(row.Recall), "f1": rounded(row.F1), "iou": rounded(row.IoU)}
                    for row in table.itertuples()]
    return splits, test_classes


def export(ckpt_path: Path, lmdb_path: Path, parquet_path: Path, out_dir: Path):
    """Predict every city of the parquet with a checkpoint and write results.json and the maps into out_dir"""
    checkpoint = f"{ckpt_path.resolve().parents[1].name}/{ckpt_path.name}"
    patches = pd.read_parquet(parquet_path, columns=["city"])["city"].value_counts().sort_index()
    overview_city = OVERVIEW_CITY if OVERVIEW_CITY in patches.index else patches.index[0]
    print(f"\nExporting {', '.join(patches.index)} with {checkpoint} to {out_dir}...")
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    model, args = load_model(ckpt_path)

    cities = []
    for city, count in patches.items():
        result = predict_city(model, lmdb_path, parquet_path, city)
        splits, test_classes = scores(result["confusion"])
        height, width = result["label"].shape
        cities.append({"name": city, "slug": city.lower(), "patches": int(count), "width": width, "height": height,
                       "splits": splits, "test_classes": test_classes,
                       "images": save_maps(city.lower(), result, out_dir)})
        if city == overview_city:
            save_overview(result, out_dir / "maps" / "overview.png")

    results = {
        "model": {"arch": args.arch_name, "description": ARCHS[args.arch_name], "pretrained": args.pretrained,
                  "epochs": args.epochs, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                  "batch_size": args.batch_size,
                  # No city filter in experiments.py means all cities of the parquet
                  "train_cities": args.train_cities or sorted(patches.keys()),
                  "test_cities": args.test_cities or sorted(patches.keys()),
                  # Checkpoints from before the --seed option have no seed
                  "seed": getattr(args, "seed", None),
                  "checkpoint": checkpoint, "exported": date.today().isoformat()},
        "classes": [{"id": k, "code": LCZ_CODES[k], "name": name, "color": "#" + COLORS[k].tobytes().hex()}
                    for k, name in enumerate(["No data"] + CLASS_NAMES[1:])],
        "cities": cities,
    }
    # allow_nan=False, browsers cannot parse NaN in JSON
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, allow_nan=False), encoding="utf-8")
    size = sum(p.stat().st_size for p in (out_dir / "maps").iterdir())
    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'maps'}, {size / 1e6:.1f} MB of maps")
