import random

import chess
import torch
from lightning import LightningModule

from chess_ai.board_tokenizer import BoardTokenizer
from utils.logger import log
from tqdm import tqdm


def move_to_index(move: chess.Move) -> int:
    """
    Converts a chess.Move into an index.

    Mapping: index = from_square * 64 + to_square.
    (Note: This simple mapping does not handle promotions.)
    """
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Converts an index back to a chess.Move given the current board state.

    Returns a legal move if possible, otherwise None.
    """
    from_square, to_square = divmod(index, 64)
    move = chess.Move(from_square, to_square)
    return move if move in board.legal_moves else None


def self_play_game(
    model: LightningModule, tokenizer: BoardTokenizer, device: torch.device
) -> list:
    """
    Plays a self-play game using the current model.

    At each turn, the model evaluates the board state and outputs a policy distribution.
    The code then masks illegal moves and samples a move from the legal ones.

    Returns:
        game_data: A list of dictionaries containing:
          - 'board': Tokenized board state (tensor).
          - 'policy_target': The chosen move index (int).
          - 'value_target': Game outcome (float).
    """
    model.eval()
    board = chess.Board()
    game_data = []

    while not board.is_game_over():
        board_tensor = tokenizer.tokenize(board).unsqueeze(0).to(device)

        with torch.no_grad():
            policy_logits, _ = model(board_tensor)
            policy_logits = policy_logits.squeeze(0)
            legal_moves = list(board.legal_moves)
            legal_indices = [move_to_index(m) for m in legal_moves]
            mask = torch.full((4096,), -1e9, device=device)
            mask[legal_indices] = 0
            masked_logits = policy_logits + mask
            probabilities = torch.softmax(masked_logits, dim=0)

        try:
            move_index = torch.multinomial(probabilities, 1).item()
        except Exception as e:
            log.error("Error sampling move: %s", e)
            move_index = random.choice(legal_indices)

        move = index_to_move(move_index, board)
        if move is None:
            move = random.choice(legal_moves)
            move_index = move_to_index(move)

        game_data.append(
            {
                "board": board_tensor.squeeze(0).cpu(),
                "policy_target": move_index,
                "value_target": None,
            }
        )

        board.push(move)

    result = board.result()
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = -1.0
    else:
        outcome = 0.0

    for i, data in enumerate(game_data):
        data["value_target"] = torch.tensor(
            [outcome if i % 2 == 0 else -outcome], dtype=torch.float
        )

    return game_data


def generate_self_play_data(
    model: LightningModule,
    tokenizer: BoardTokenizer,
    device: torch.device,
    num_games: int = 10,
) -> list:
    """
    Generates self-play data by having the current model play a number of games.

    Returns:
        A list of data entries from all games.
    """
    all_data = []
    for game_idx in tqdm(
        range(num_games), desc="Generating self-play games", total=num_games
    ):
        game_data = self_play_game(model, tokenizer, device)
        all_data.extend(game_data)
    return all_data
