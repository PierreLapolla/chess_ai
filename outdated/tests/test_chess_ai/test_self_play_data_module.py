import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler
from chess_ai.self_play_data_module import SelfPlayDataset, SelfPlayDataModule


@pytest.fixture
def sample_data():
    data = []
    for i in range(10):
        board_tensor = torch.full((64,), i, dtype=torch.float)
        sample = {
            "board": board_tensor,
            "policy_target": i,
            "value_target": torch.tensor(
                [1.0 if i % 2 == 0 else -1.0], dtype=torch.float
            ),
        }
        data.append(sample)
    return data


def test_dataset_length(sample_data):
    dataset = SelfPlayDataset(sample_data)
    assert len(dataset) == len(sample_data)


def test_dataset_getitem(sample_data):
    dataset = SelfPlayDataset(sample_data)
    for i, expected in enumerate(sample_data):
        item = dataset[i]
        assert torch.equal(item["board"], expected["board"])
        assert item["policy_target"] == expected["policy_target"]
        assert torch.equal(item["value_target"], expected["value_target"])


def test_datamodule_setup_and_dataloader(sample_data):
    batch_size = 4
    datamodule = SelfPlayDataModule(sample_data, batch_size=batch_size)
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    assert isinstance(dataloader, DataLoader)
    batch = next(iter(dataloader))
    for key in ["board", "policy_target", "value_target"]:
        assert key in batch
    assert batch["board"].dim() == 2
    assert batch["board"].shape[1] == 64
    assert batch["policy_target"].dim() == 1
    assert batch["policy_target"].shape[0] <= batch_size
    assert batch["value_target"].dim() == 2
    assert batch["value_target"].shape[1] == 1


def test_dataloader_shuffle_property(sample_data):
    batch_size = 3
    datamodule = SelfPlayDataModule(sample_data, batch_size=batch_size)
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    assert isinstance(dataloader.sampler, RandomSampler)
