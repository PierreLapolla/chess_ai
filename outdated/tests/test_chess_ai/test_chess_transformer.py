import torch
import pytest
from torch.optim import Optimizer
from chess_ai.chess_transformer import ChessTransformer


@pytest.fixture
def model():
    return ChessTransformer()


@pytest.fixture
def dummy_batch():
    batch_size = 4
    boards = torch.randint(0, 13, (batch_size, 64))
    policy_target = torch.randint(0, 4096, (batch_size,))
    value_target = torch.randn(batch_size, 1)
    return {
        "board": boards,
        "policy_target": policy_target,
        "value_target": value_target,
    }


def test_forward_shapes(model):
    batch_size = 4
    boards = torch.randint(0, 13, (batch_size, 64))
    policy_logits, value = model(boards)
    assert policy_logits.shape == (batch_size, 4096)
    assert value.shape == (batch_size, 1)


def test_training_step_returns_scalar_loss(model, dummy_batch):
    loss = model.training_step(dummy_batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert loss.dim() == 0


def test_configure_optimizers_returns_optimizer(model):
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, Optimizer)


def test_training_step_no_nan(model, dummy_batch):
    loss = model.training_step(dummy_batch, batch_idx=0)
    assert not torch.isnan(loss)


def test_loss_backward(model, dummy_batch):
    loss = model.training_step(dummy_batch, batch_idx=0)
    loss.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
