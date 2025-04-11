from abc import abstractmethod
from datetime import datetime
from typing import Any

import torch
import wandb
from lightning import LightningModule
from torch.optim import Adam

from utils.logger import log
from lightning_template.config import config


class BaseModel(LightningModule):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.loss_func = self.get_loss_func()
        self.save_hyperparameters()

    @abstractmethod
    def get_loss_func(self):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=config.learning_rate)
        return optimizer

    def _step(self, batch, loss_name: str) -> Any:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.log(loss_name, loss)
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        return self._step(batch, "val_loss")

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        return self._step(batch, "test_loss")

    def on_train_epoch_start(self) -> None:
        self.log(
            "lr", self.optimizers().param_groups[0]["lr"], prog_bar=True, logger=True
        )

    def on_train_end(self) -> None:
        path = config.root_path / "models" / self.__class__.__name__
        path.mkdir(parents=True, exist_ok=True)
        model_name = f"model_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pt"
        torch.save(self.state_dict(), path / model_name)
        wandb.save(str(path / model_name))
        log.info(f"Saved model to {path / model_name}")
