from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SelfPlayDataset(Dataset):
    """
    Dataset to wrap self-play generated chess data.

    Each item is a dictionary containing:
        - 'board': Tensor (64,) representing tokenized board state.
        - 'policy_target': Integer move index.
        - 'value_target': Tensor (1,) representing the game outcome.
    """

    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SelfPlayDataModule(LightningDataModule):
    def __init__(self, data: list, batch_size: int = 8):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.train_dataset = None

    def setup(self, stage: str = None):
        self.train_dataset = SelfPlayDataset(self.data)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=7, shuffle=True
        )
