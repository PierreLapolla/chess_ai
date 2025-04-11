from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from utils.logger import log
from pathlib import Path
from lightning_template.config import config


class DataModule(LightningDataModule):
    def __init__(self) -> None:
        super(DataModule, self).__init__()
        self.data_path = Path(__file__).parent.parent.parent / "data"
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        log.info(f"Using {self.num_workers} workers for data loading.")

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.predict_set = None

    def prepare_data(self) -> None:
        MNIST(self.data_path, train=True, download=True)
        MNIST(self.data_path, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            mnist_full = MNIST(self.data_path, train=True, transform=ToTensor())
            self.train_set, self.val_set = random_split(mnist_full, [55000, 5000])

        elif stage == "test":
            self.test_set = MNIST(self.data_path, train=False, transform=ToTensor())

        elif stage == "predict":
            self.predict_set = MNIST(self.data_path, train=False, transform=ToTensor())

        else:
            message = f"Stage '{stage}' is not recognized."
            log.error(message)
            raise ValueError(message)

    def _get_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_set)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_set)

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.predict_set)
