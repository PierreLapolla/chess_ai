from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from lightning_template.config import config


def get_trainer() -> Trainer:
    logger = WandbLogger(
        project="template_project",
        entity="deldrel",
        save_dir=str(config.root_path / "logs"),
    )

    return Trainer(
        max_epochs=config.max_epochs,
        callbacks=[],
        logger=logger,
    )
