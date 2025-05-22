import pandas as pd
import lmdb
import safetensors.numpy as stnp
import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule


class MLCZIndexableLMDBDataset(Dataset):
    def __init__(self, lmdb_path, metadata_parquet_path, split, transform=None):
        """
        Dataset for the MLCZ Task using a lmdb file.

        :param lmdb_path: path to the lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        self.lmdb_path = lmdb_path
        self.metadata = pd.read_parquet(metadata_parquet_path)
        self.split = split
        self.transform = transform

        if self.split:
            self.metadata = self.metadata[self.metadata['split'] == self.split]

        self.env = None

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
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N, )
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


class MLCZDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            base_path: Optional[str] = None,
            lmdb_path: Optional[str] = None,
            metadata_parquet_path: Optional[str] = None,
            transform=None,
    ):
        """
        DataModule for the MLCZ Task dataset.

        :param batch_size: batch size for the dataloaders
        :param num_workers: number of workers for the dataloaders
        :param base_path: path to the source BigEarthNet dataset (root of the tar file), for tif dataset
        :param lmdb_path: path to the converted lmdb file, for lmdb dataset
        :param metadata_parquet_path: path to the metadata parquet file, for lmdb dataset
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.base_path = base_path
        self.lmdb_path = lmdb_path
        self.metadata_parquet_path = metadata_parquet_path
        self.transform = transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Set up datasets for training, validation, and testing splits.
        """
        dataset_class = None
        dataset_args = {}

        dataset_class = MLCZIndexableLMDBDataset
        dataset_args = {
            'lmdb_path': self.lmdb_path,
            'metadata_parquet_path': self.metadata_parquet_path
        }
        self.train_dataset = dataset_class(**dataset_args, split='train')
        self.val_dataset = dataset_class(**dataset_args, split='validation')
        self.test_dataset = dataset_class(**dataset_args, split='test')

    def train_dataloader(self):
        """
        Return DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True
        )

    def val_dataloader(self):
        """
        Return DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False
        )

    def test_dataloader(self):
        """
        Return DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False
        )