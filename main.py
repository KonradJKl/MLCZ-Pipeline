from modules import convert_data
import subprocess
from pathlib import Path
import itertools
import wandb
import os
import random
import numpy as np
import torch
import lightning.pytorch as pl
from dotenv import load_dotenv

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
wandb.login(key=os.getenv("WANDB_API_KEY"))
os.environ['SRC'] = "C:\\Users\\Konrad\\PycharmProjects\\MLCZ-Pipeline\\data"
os.environ['LMDB'] = "C:\\Users\\Konrad\\PycharmProjects\\MLCZ-Pipeline\\untracked-files\\MLCZ.lmdb"
os.environ['PARQUET'] = "C:\\Users\\Konrad\\PycharmProjects\\MLCZ-Pipeline\\untracked-files\\MLCZ.parquet"


def build_command(base_cmd, **kwargs):
    cmd = base_cmd.copy()
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(command):
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

    if result.returncode == 0:
        print("Training completed successfully.")
    else:
        print("Error occurred during training:")

    print("Training Output:\n", result.stdout)
    print("Error Log:\n", result.stderr)


def architecture_experiments(base_command, base_path):  # **Architecture Experiments**
    arch_names = ["ResNet18", "CustomCNN", "ViT-Tiny", "ConvNeXt-Nano"]
    pretrained_options = [True, False]
    dropout_options = [True, False]
    datasets = ["MLCZ"]
    for arch_name, dataset, pretrained, dropout in itertools.product(arch_names, datasets, pretrained_options, dropout_options):
        print(f"Running experiment: arch_name={arch_name}, pretrained={pretrained}, dropout={dropout}, dataset={dataset}")

        resize = (arch_name == "ViT-Tiny")
        channels, classes = (12, 19)  # TODO: Update

        lmdb_path = base_path / f"{dataset}.lmdb"
        parquet_path = base_path / f"{dataset}.parquet"

        command = build_command(base_command,
                                arch_name=arch_name,
                                dataset=dataset,
                                lmdb_path=str(lmdb_path),
                                metadata_parquet_path=str(parquet_path),
                                resize=resize,
                                num_channels=channels,
                                num_classes=classes,
                                learning_rate=0.001,
                                pretrained=pretrained,
                                dropout=dropout)
        run_experiment(command)


def run_training():
    seed_everything(42)
    base_path = Path.cwd() / "untracked-files"
    lmdb_path = base_path / "MLCZ.lmdb"  # untracked-files/EuroSAT.lmdb
    parquet_path = base_path / "MLCZ.parquet"  # untracked-files/EuroSAT.parquet
    logs_dir = Path.cwd() / "logs"  # "attachments" /
    logs_dir.mkdir(parents=True, exist_ok=True)

    assert lmdb_path.exists(), f"LMDB path does not exist: {lmdb_path}"
    assert parquet_path.exists(), f"Parquet path does not exist: {parquet_path}"

    base_command = [
        "python", "modules/experiments.py",  # "poetry", "run", "python"
        "--logging_dir", str(logs_dir),
        "--logger", "wandb",
        "--num_workers", "4",
        "--batch_size", "32",
        "--epochs", "30",
        "--weight_decay", "0.01",
    ]
    action_set = ["architecture"]
    if "architecture" in action_set:
        # ** Architecture Experiments **
        architecture_experiments(base_command, base_path)


if __name__ == '__main__':

    print("\nConverting Data")
    convert_data.load_data(input_data_path=os.environ['SRC'],
                           output_lmdb_path=os.environ['LMDB'],
                           output_parquet_path=os.environ['PARQUET'])
    print("\nData conversion completed")

    #run_training()