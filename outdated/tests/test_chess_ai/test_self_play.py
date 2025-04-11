import chess
import pytest
import torch
from lightning import LightningModule

from chess_ai.self_play import (
    move_to_index,
    index_to_move,
    self_play_game,
    generate_self_play_data,
)
from utils.logger import log


class DummyModel(LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, board_tensor):
        return torch.zeros(4096, device=board_tensor.device), torch.tensor(
            0.0, device=board_tensor.device
        )

    def eval(self):
        pass


class DummyTokenizer:
    def tokenize(self, board: chess.Board):
        return torch.zeros(64)


@pytest.fixture
def device():
    return torch.device("cpu")


def test_move_to_index_and_index_to_move():
    board = chess.Board()
    legal_moves = list(board.legal_moves)
    move = legal_moves[0]
    index = move_to_index(move)
    result_move = index_to_move(index, board)
    assert result_move == move
    illegal_index = 4095
    result_illegal_move = index_to_move(illegal_index, board)
    assert result_illegal_move is None


def test_edge_case_promotion():
    board = chess.Board("7k/2P5/8/8/8/8/8/7K w - - 0 1")
    promotion_move = None
    for move in board.legal_moves:
        if move.promotion is not None:
            promotion_move = move
            break
    if promotion_move:
        index = move_to_index(promotion_move)
        result_move = index_to_move(index, board)
        assert result_move is None or result_move != promotion_move


def test_self_play_game_structure(device):
    model = DummyModel()
    tokenizer = DummyTokenizer()
    game_data = self_play_game(model, tokenizer, device)
    assert game_data
    for entry in game_data:
        assert "board" in entry
        assert "policy_target" in entry
        assert "value_target" in entry
        assert isinstance(entry["board"], torch.Tensor)
        assert entry["board"].nelement() == 64
        assert isinstance(entry["policy_target"], int)
        assert isinstance(entry["value_target"], torch.Tensor)
        assert entry["value_target"].numel() == 1


def test_sampling_error(monkeypatch, device):
    model = DummyModel()
    tokenizer = DummyTokenizer()

    def failing_multinomial(probabilities, num, **kwargs):
        raise Exception("Forced error")

    monkeypatch.setattr(torch, "multinomial", failing_multinomial)
    monkeypatch.setattr(log, "error", lambda *args, **kwargs: None)
    game_data = self_play_game(model, tokenizer, device)
    assert game_data
    for entry in game_data:
        assert 0 <= entry["policy_target"] < 4096


def test_generate_self_play_data(device):
    model = DummyModel()
    tokenizer = DummyTokenizer()
    num_games = 2

    data = generate_self_play_data(model, tokenizer, device, num_games=num_games)
    assert isinstance(data, list)
    assert len(data) > 0

    for entry in data:
        assert "board" in entry
        assert "policy_target" in entry
        assert "value_target" in entry
        assert isinstance(entry["board"], torch.Tensor)
        assert entry["board"].nelement() == 64
        assert isinstance(entry["policy_target"], int)
        assert isinstance(entry["value_target"], torch.Tensor)
        assert entry["value_target"].numel() == 1
