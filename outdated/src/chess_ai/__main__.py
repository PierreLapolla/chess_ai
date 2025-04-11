from lightning import seed_everything

from chess_ai.chess_bot import train_loop
from utils.timer import timer


@timer
def main() -> None:
    seed_everything(42)
    train_loop()


if __name__ == "__main__":
    main()
