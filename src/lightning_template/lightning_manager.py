import wandb

from lightning_template.data_module import DataModule
from lightning_template.mnist_model import MNISTModel
from lightning_template.trainer import get_trainer
from utils.singleton import singleton
from utils.timer import timer
from utils.try_except import try_except


@singleton
class LightningManager:
    def __init__(self):
        self.data_module = DataModule()
        self.model = MNISTModel()
        self.trainer = get_trainer()

    @timer
    @try_except(finally_callable=wandb.finish)
    def start_training(self) -> None:
        self.trainer.fit(self.model, self.data_module)
        self.trainer.test(self.model, self.data_module)
