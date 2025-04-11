from pathlib import Path

import torch
from lightning import Trainer

from chess_ai.self_play_data_module import SelfPlayDataModule
from chess_ai.board_tokenizer import BoardTokenizer
from chess_ai.chess_transformer import ChessTransformer
from chess_ai.self_play import generate_self_play_data
from utils.logger import log


def train_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    tokenizer = BoardTokenizer()
    model = ChessTransformer()
    model.to(device)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    self_play_data = generate_self_play_data(model, tokenizer, device, num_games=5)
    data_module = SelfPlayDataModule(self_play_data, batch_size=8)

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=str(checkpoint_dir),
        log_every_n_steps=1,
        enable_checkpointing=True,
    )

    trainer.fit(model, data_module)
