import pandas as pd
import lmdb
import safetensors.numpy as stnp
import torch
from typing import Optional, List, Union
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule


class MLCZIndexableLMDBDataset(Dataset):
    def __init__(self, lmdb_path, metadata_parquet_path, split, transform=None,
                 cities=None, label_filter=None, min_label_diversity=None):
        """
        Dataset for the MLCZ Task using a lmdb file.

        :param lmdb_path: path to the lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        :param cities: list of cities to include, None for all cities
        :param label_filter: list of dominant labels to include, None for all labels
        :param min_label_diversity: minimum number of different labels per patch
        """
        self.lmdb_path = lmdb_path
        self.metadata = pd.read_parquet(metadata_parquet_path)
        self.split = split
        self.transform = transform

        # Apply filters
        if self.split:
            self.metadata = self.metadata[self.metadata['split'] == self.split]

        if cities is not None:
            if isinstance(cities, str):
                cities = [cities]
            self.metadata = self.metadata[self.metadata['city'].isin(cities)]
            print(f"Filtered to cities: {cities} - {len(self.metadata)} patches")

        if label_filter is not None:
            if 'dominant_label' in self.metadata.columns:
                self.metadata = self.metadata[self.metadata['dominant_label'].isin(label_filter)]
                print(f"Filtered to dominant labels: {label_filter} - {len(self.metadata)} patches")

        if min_label_diversity is not None:
            if 'label_diversity' in self.metadata.columns:
                self.metadata = self.metadata[self.metadata['label_diversity'] >= min_label_diversity]
                print(f"Filtered to min diversity {min_label_diversity} - {len(self.metadata)} patches")

        if len(self.metadata) == 0:
            raise ValueError("No patches match the specified filters!")

        self.env = None

        # Store available cities and labels for reference
        self.available_cities = sorted(self.metadata['city'].unique())
        if 'dominant_label' in self.metadata.columns:
            self.available_labels = sorted(self.metadata['dominant_label'].unique())
        else:
            self.available_labels = None

    def _init_env(self):
        """
        Initialize the LMDB environment.
        :return: None
        """
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        """
        Get the number of items in the dataset.
        :return: number of items in the dataset
        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: index of the item to get
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (H, W)
        """
        self._init_env()
        sample_metadata = self.metadata.iloc[idx]

        with self.env.begin() as txn:
            img_key = sample_metadata['patch_id']
            img_data = txn.get(img_key.encode())
            tensors = stnp.load(img_data)

        image = torch.Tensor(tensors['data'])

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(tensors['label'])

        return image, labels

    def get_dataset_info(self):
        """Get information about the current dataset."""
        info = {
            'total_patches': len(self.metadata),
            'cities': self.available_cities,
            'city_counts': dict(self.metadata['city'].value_counts()),
        }

        if 'dominant_label' in self.metadata.columns:
            info.update({
                'available_labels': self.available_labels,
                'label_counts': dict(self.metadata['dominant_label'].value_counts()),
                'avg_label_diversity': self.metadata['label_diversity'].mean() if 'label_diversity' in self.metadata.columns else None
            })

        return info


class MLCZDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            base_path: Optional[str] = None,
            lmdb_path: Optional[str] = None,
            metadata_parquet_path: Optional[str] = None,
            transform=None,
            train_cities: Optional[Union[str, List[str]]] = None,
            val_cities: Optional[Union[str, List[str]]] = None,
            test_cities: Optional[Union[str, List[str]]] = None,
            label_filter: Optional[List[int]] = None,
            min_label_diversity: Optional[int] = None,
    ):
        """
        DataModule for the MLCZ Task dataset with flexible city and label selection.

        :param batch_size: batch size for the dataloaders
        :param num_workers: number of workers for the dataloaders
        :param base_path: path to the source dataset (for future extensions)
        :param lmdb_path: path to the converted lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param transform: transform to apply to images
        :param train_cities: cities to use for training (None = all available)
        :param val_cities: cities to use for validation (None = all available)
        :param test_cities: cities to use for testing (None = all available)
        :param label_filter: list of dominant labels to include (None = all labels)
        :param min_label_diversity: minimum label diversity per patch (None = no filter)
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.base_path = base_path
        self.lmdb_path = lmdb_path
        self.metadata_parquet_path = metadata_parquet_path
        self.transform = transform

        # City selection
        self.train_cities = train_cities
        self.val_cities = val_cities
        self.test_cities = test_cities

        # Label filtering
        self.label_filter = label_filter
        self.min_label_diversity = min_label_diversity

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Will be populated during setup
        self.classes = None
        self.available_cities = None

    def setup(self, stage=None):
        """
        Set up datasets for training, validation, and testing splits.
        """
        print(f"\n🏗️  Setting up MLCZ DataModule...")

        # Read metadata to get available cities and classes
        metadata = pd.read_parquet(self.metadata_parquet_path)
        self.available_cities = sorted(metadata['city'].unique())

        if 'dominant_label' in metadata.columns:
            self.classes = sorted(metadata['dominant_label'].unique())

        print(f"Available cities: {self.available_cities}")
        if self.classes:
            print(f"Available classes: {self.classes}")

        dataset_args = {
            'lmdb_path': self.lmdb_path,
            'metadata_parquet_path': self.metadata_parquet_path,
            'transform': self.transform,
            'label_filter': self.label_filter,
            'min_label_diversity': self.min_label_diversity
        }

        # Create datasets with city filtering
        self.train_dataset = MLCZIndexableLMDBDataset(
            **dataset_args,
            split='train',
            cities=self.train_cities
        )

        self.val_dataset = MLCZIndexableLMDBDataset(
            **dataset_args,
            split='validation',
            cities=self.val_cities
        )

        self.test_dataset = MLCZIndexableLMDBDataset(
            **dataset_args,
            split='test',
            cities=self.test_cities
        )

        # Print dataset information
        print(f"\n📊 Dataset Statistics:")
        for name, dataset in [('Train', self.train_dataset), ('Val', self.val_dataset), ('Test', self.test_dataset)]:
            info = dataset.get_dataset_info()
            print(f"  {name}: {info['total_patches']} patches from cities {info['cities']}")
            if 'label_counts' in info:
                print(f"    Label distribution: {info['label_counts']}")

    def train_dataloader(self):
        """Return DataLoader for the training dataset."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True
        )

    def val_dataloader(self):
        """Return DataLoader for the validation dataset."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False
        )

    def test_dataloader(self):
        """Return DataLoader for the test dataset."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False
        )

    @classmethod
    def create_for_single_city(cls, city_name: str, **kwargs):
        """
        Convenience method to create a datamodule for a single city.

        :param city_name: Name of the city to use
        :param kwargs: Other arguments for MLCZDataModule
        :return: MLCZDataModule instance
        """
        return cls(
            train_cities=[city_name],
            val_cities=[city_name],
            test_cities=[city_name],
            **kwargs
        )

    @classmethod
    def create_for_cross_city_evaluation(cls, train_cities: List[str], test_city: str, **kwargs):
        """
        Convenience method for cross-city domain adaptation experiments.

        :param train_cities: Cities to use for training
        :param test_city: City to use for testing
        :param kwargs: Other arguments for MLCZDataModule
        :return: MLCZDataModule instance
        """
        return cls(
            train_cities=train_cities,
            val_cities=train_cities,  # Use training cities for validation
            test_cities=[test_city],
            **kwargs
        )

        total_sum = None
        total_sq_sum = None
        total_pixels = 0

        for images, _ in loader:
            # images: Tensor of shape (B, C, H, W)
            B, C, H, W = images.shape
            pixels = B * H * W

            # lazily init accumulators once we know C
            if total_sum is None:
                total_sum = torch.zeros(C)
                total_sq_sum = torch.zeros(C)

            # sum over batch and spatial dims
            sum_    = images.sum(dim=[0, 2, 3])          # shape (C,)
            sq_sum  = (images ** 2).sum(dim=[0, 2, 3])    # shape (C,)

            total_sum    += sum_
            total_sq_sum += sq_sum
            total_pixels += pixels

        # compute mean & std
        means = total_sum / total_pixels
        stds  = torch.sqrt(total_sq_sum / total_pixels - means ** 2)

        # clean up LMDB env to avoid leaking file handles
        if hasattr(dataset, 'env') and dataset.env is not None:
            dataset.env.close()
            dataset.env = None

        # return as numpy arrays (so you can feed them into transforms.Normalize)
        return means.numpy(), stds.numpy()