import concurrent.futures
import random

import chess
from datasets import load_dataset
from torch.utils.data import IterableDataset, Dataset

from lightning_template.base_model import BaseModel
from lightning_template.board_tokenizer import BoardTokenizer
from lightning_template.config import config
from utils.try_except import try_except


class SelfPlayIterableDataset(IterableDataset):
    def __init__(self, model: BaseModel, buffer_size=100):
        self.model = model
        self.openings_dataset = self.load_openings()
        self.tokenizer = BoardTokenizer()
        self.num_workers = config.self_play_num_workers
        self.buffer_size = buffer_size

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        )
        self.futures = []

    @try_except(error_callable=exit)
    def load_openings(self) -> Dataset:
        return load_dataset("Lichess/chess-openings", split="train")

    def run_self_play_game(self, starting_moves: str) -> dict:
        board = chess.Board(fen=starting_moves)
        while not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            board.push(move)

        dummy_tensor = self.tokenizer.tokenize(board)
        return {"final_board": dummy_tensor}

    @try_except
    def __iter__(self):
        while True:
            while len(self.futures) < self.buffer_size:
                opening = self.openings_dataset[
                    random.randrange(len(self.openings_dataset))
                ]
                starting_moves = opening.get("epd", [])

                future = self.executor.submit(self.run_self_play_game, starting_moves)
                self.futures.append(future)

            done, self.futures = concurrent.futures.wait(
                self.futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in list(done):
                result = future.result()
                yield result

    def shutdown(self):
        self.executor.shutdown()
