import argparse
import torch
import os
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms
from base import BaseModel
from model import get_network
from visualiztion import LCZVisualizer, visualize_model_predictions
from MLCZ import MLCZDataModule

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
parser = argparse.ArgumentParser(prog='MLCZ-Pipeline', description='Run Experiments.')

parser.add_argument('--logging_dir', type=str)
parser.add_argument("--logger", type=str, default="wandb")

parser.add_argument('--dataset', default='MLCZ', type=str, required=True)
parser.add_argument('--lmdb_path', type=str)
parser.add_argument('--metadata_parquet_path', type=str)

parser.add_argument('--num_channels', type=int, default=18, required=True)
parser.add_argument('--num_classes', type=int, default=244, required=True)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--arch_name', type=str, choices=["unet", "CustomCNN"], required=True)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--augmentation', type=str, default=None)
parser.add_argument('--class_weights', action='store_true')
parser.add_argument('--train_cities', type=str, nargs='*', default=None, help='Cities to use for training')
parser.add_argument('--val_cities', type=str, nargs='*', default=None, help='Cities to use for validation')
parser.add_argument('--test_cities', type=str, nargs='*', default=None, help='Cities to use for testing')
parser.add_argument('--label_filter', type=int, nargs='*', default=None, help='Label IDs to include')
parser.add_argument('--min_label_diversity', default=None, help='Minimum label diversity per patch')


def run_benchmark(args, arch_name, pretrained, dropout, dataset, logger):
    if dataset == "MLCZ":
        datamodule = MLCZDataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lmdb_path=args.lmdb_path,
            metadata_parquet_path=args.metadata_parquet_path,
            base_path=None,
            train_cities=args.train_cities,
            val_cities=args.val_cities,
            test_cities=args.test_cities,
            label_filter=args.label_filter,
            min_label_diversity=args.min_label_diversity
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    datamodule.setup()

    network = get_network(
        arch_name=arch_name,
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou_macro",
        save_top_k=1,
        mode="max"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_iou_macro",
        patience=15,
        mode="max",
        verbose=True,
        min_delta=0.001
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=logger,
        deterministic="warn",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )

    model = BaseModel(
        args=args,
        datamodule=datamodule,
        network=network
    )

    trainer.fit(model)
    test_results = trainer.test(model, ckpt_path="best")

    # Generate visualizations if requested

    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)

    # Create visualizer
    visualizer = LCZVisualizer(save_dir=os.path.join(args.logging_dir, "visualizations"))

    # Generate visualizations on test set
    experiment_name = f"{args.dataset}_{args.arch_name}_pt={args.pretrained}_do={args.dropout}"

    # Load best checkpoint for visualization
    best_model = BaseModel.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        args=args,
        datamodule=datamodule,
        network=network
    )

    metrics = visualize_model_predictions(
        best_model,
        datamodule.test_dataloader(),
        visualizer,
        experiment_name,
        device=torch.cuda.get_device_name(0),
        max_batches=20  # Process more batches for better statistics
    )

    # Log metrics to wandb if using wandb logger
    if args.logger == "wandb" and logger:
        logger.log_metrics({
            "test/accuracy": metrics['accuracy'],
            "test/macro_f1": metrics['macro_f1'],
            "test/weighted_f1": metrics['weighted_f1'],
            "test/macro_precision": metrics['macro_precision'],
            "test/macro_recall": metrics['macro_recall']
        })

    return test_results


if __name__ == "__main__":
    arguments = parser.parse_args()
    print(arguments)
    logger = WandbLogger(
        project="MLCZ-Pipeline-Server",
        save_dir=arguments.logging_dir,
        group=arguments.dataset,
        name=f"{arguments.dataset}_{arguments.arch_name}_pt={arguments.pretrained}_do={arguments.dropout}" if arguments.augmentation is None else f"{arguments.dataset}_{arguments.augmentation}"
    )
    run_benchmark(arguments, arguments.arch_name, arguments.pretrained, arguments.dropout, arguments.dataset, logger)