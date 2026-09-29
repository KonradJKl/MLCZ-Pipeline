from modules import convert_data, static_page
from modules.city_maps import find_checkpoints
from modules.config import ROOT, DATA_DIR, OUTPUT_DIR, LOGS_DIR, DOCS_DIR, LMDB_PATH, PARQUET_PATH
import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(prog='MLCZ-Pipeline', description='Convert the rasters and run the experiments.')
parser.add_argument('--convert', action='store_true', help='convert the rasters in DATA_DIR into an LMDB and a parquet file')
parser.add_argument('--train', action='store_true', help='run the experiments on the converted data')
parser.add_argument('--export', action='store_true', help='write results.json and the city maps into DOCS_DIR')
parser.add_argument('--checkpoint', type=Path, help='checkpoint for --export, default: the newest one in LOGS_DIR')
parser.add_argument('--stratify_by', type=str, default='mixed', choices=['city', 'labels', 'mixed', 'none'])
parser.add_argument('--split_strategy', type=str, default='hybrid', choices=['city_based', 'patch_based', 'hybrid'])
parser.add_argument('--map_size', type=float, default=4e10, help='LMDB size in bytes, reserved on disk right away on Windows')
parser.add_argument('--epochs', type=int, default=50)


def build_command(base_cmd, **kwargs):
    cmd = base_cmd.copy()
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def run_experiment(command):
    # Flush first, otherwise our prints end up after the child's output when stdout is a file
    print(f"Executing: {' '.join(command)}", flush=True)
    return subprocess.run(command).returncode


def architecture_experiments(base_command):
    # Pretrained weights only exist for the U-Net encoder and only the CustomCNN uses dropout
    experiments = [("unet", True, False), ("unet", False, False), ("CustomCNN", False, True), ("CustomCNN", False, False)]
    failed = []
    for arch_name, pretrained, dropout in experiments:
        print(f"Running experiment: arch_name={arch_name}, pretrained={pretrained}, dropout={dropout}")
        command = build_command(base_command, arch_name=arch_name, pretrained=pretrained, dropout=dropout)
        if run_experiment(command) != 0:
            failed.append(f"{arch_name} (pretrained={pretrained}, dropout={dropout})")
    return failed


def run_conversion(stratify_by: str, split_strategy: str, map_size: float):
    if LMDB_PATH.exists() and PARQUET_PATH.exists():
        print(f"\n{LMDB_PATH} and {PARQUET_PATH} already exist, delete them to convert again")
        return

    print("\nConverting Data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_data.load_data(input_data_path=str(DATA_DIR),
                           output_lmdb_path=str(LMDB_PATH),
                           output_parquet_path=str(PARQUET_PATH),
                           stratify_by=stratify_by,
                           split_strategy=split_strategy,
                           map_size=map_size)
    print("\nData conversion completed")


def run_training(epochs: int):
    # Seeding happens in experiments.py, every run is its own process
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    assert LMDB_PATH.exists(), f"LMDB path does not exist: {LMDB_PATH}"
    assert PARQUET_PATH.exists(), f"Parquet path does not exist: {PARQUET_PATH}"

    # Hyperparameters from the final presentation (June 2025)
    base_command = [
        sys.executable, str(ROOT / "modules" / "experiments.py"),
        "--logging_dir", str(LOGS_DIR),
        "--lmdb_path", str(LMDB_PATH),
        "--metadata_parquet_path", str(PARQUET_PATH),
        "--num_workers", "4",
        "--batch_size", "32",
        "--epochs", str(epochs),
        "--learning_rate", "0.0005",
        "--weight_decay", "0.0001",
    ]
    return architecture_experiments(base_command)


def run_export(checkpoint: Path):
    assert LMDB_PATH.exists(), f"LMDB path does not exist: {LMDB_PATH}"
    assert PARQUET_PATH.exists(), f"Parquet path does not exist: {PARQUET_PATH}"
    checkpoints = [checkpoint] if checkpoint else find_checkpoints(LOGS_DIR)
    assert checkpoints, f"No checkpoint found in {LOGS_DIR}, run --train first or pass --checkpoint"
    static_page.export(checkpoints[0], LMDB_PATH, PARQUET_PATH, DOCS_DIR)


if __name__ == '__main__':
    args = parser.parse_args()
    if not (args.convert or args.train or args.export):
        parser.error("nothing to do, use --convert, --train and/or --export")

    if args.convert:
        run_conversion(args.stratify_by, args.split_strategy, args.map_size)
    if args.train:
        failed = run_training(args.epochs)
        if failed:
            print(f"\n{len(failed)} experiments failed: {', '.join(failed)}")
            sys.exit(1)
    if args.export:
        run_export(args.checkpoint)
