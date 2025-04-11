import chess
import pytest
import torch

from chess_ai.board_tokenizer import BoardTokenizer


def test_tokenize_standard_board():
    board = chess.Board()
    tokenizer = BoardTokenizer()
    tokens = tokenizer.tokenize(board)
    expected = torch.tensor(
        [
            4,
            2,
            3,
            5,
            6,
            3,
            2,
            4,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            10,
            8,
            9,
            11,
            12,
            9,
            8,
            10,
        ],
        dtype=torch.long,
    )
    assert torch.equal(tokens, expected)


def test_tokenize_empty_board():
    board = chess.Board()
    board.clear()
    tokenizer = BoardTokenizer()
    tokens = tokenizer.tokenize(board)
    assert tokens.shape == (64,)
    assert torch.all(tokens == torch.zeros(64, dtype=torch.long))


def test_tokenize_single_piece():
    board = chess.Board()
    board.clear()
    board.set_piece_at(28, chess.Piece(chess.KNIGHT, chess.WHITE))
    tokenizer = BoardTokenizer()
    tokens = tokenizer.tokenize(board)
    expected = torch.zeros(64, dtype=torch.long)
    expected[28] = 2
    assert torch.equal(tokens, expected)


def test_tokenize_all_pieces():
    board = chess.Board()
    board.clear()
    placements = {
        0: chess.Piece(chess.PAWN, chess.WHITE),
        1: chess.Piece(chess.KNIGHT, chess.WHITE),
        2: chess.Piece(chess.BISHOP, chess.WHITE),
        3: chess.Piece(chess.ROOK, chess.WHITE),
        4: chess.Piece(chess.QUEEN, chess.WHITE),
        5: chess.Piece(chess.KING, chess.WHITE),
        6: chess.Piece(chess.PAWN, chess.BLACK),
        7: chess.Piece(chess.KNIGHT, chess.BLACK),
        8: chess.Piece(chess.BISHOP, chess.BLACK),
        9: chess.Piece(chess.ROOK, chess.BLACK),
        10: chess.Piece(chess.QUEEN, chess.BLACK),
        11: chess.Piece(chess.KING, chess.BLACK),
    }
    for square, piece in placements.items():
        board.set_piece_at(square, piece)
    tokenizer = BoardTokenizer()
    tokens = tokenizer.tokenize(board)
    mapping = {
        chess.Piece(chess.PAWN, chess.WHITE): 1,
        chess.Piece(chess.KNIGHT, chess.WHITE): 2,
        chess.Piece(chess.BISHOP, chess.WHITE): 3,
        chess.Piece(chess.ROOK, chess.WHITE): 4,
        chess.Piece(chess.QUEEN, chess.WHITE): 5,
        chess.Piece(chess.KING, chess.WHITE): 6,
        chess.Piece(chess.PAWN, chess.BLACK): 7,
        chess.Piece(chess.KNIGHT, chess.BLACK): 8,
        chess.Piece(chess.BISHOP, chess.BLACK): 9,
        chess.Piece(chess.ROOK, chess.BLACK): 10,
        chess.Piece(chess.QUEEN, chess.BLACK): 11,
        chess.Piece(chess.KING, chess.BLACK): 12,
    }
    for square, piece in placements.items():
        assert tokens[square].item() == mapping[piece]


def test_tokenize_invalid_board():
    tokenizer = BoardTokenizer()
    with pytest.raises(AttributeError):
        tokenizer.tokenize("not a board")
