import os
from pathlib import Path
from typing import Optional

from pydantic import Field, conint, confloat
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
    PydanticBaseSettingsSource,
)

from utils.singleton import singleton
from utils.try_except import try_except


@singleton
class AppConfig(BaseSettings):
    root_path: Path = Path(__file__).parent.parent.parent
    seed: int = Field(description="Seed for random number generation")

    # Base model
    learning_rate: confloat(gt=0) = Field(description="Learning rate")

    # Data module
    batch_size: conint(gt=0) = Field(description="Batch size for data loading")
    num_workers: conint(gt=0, le=(os.cpu_count() or 1)) = Field(
        description="Number of workers for data loading"
    )
    self_play_num_workers: conint(gt=0, le=(os.cpu_count() or 1)) = Field(
        description="Number of self-play workers"
    )

    # Lightning manager

    # Trainer
    max_epochs: conint(gt=0) = Field(
        description="Maximum number of epochs for training"
    )

    # .env
    WANDB_API_KEY: Optional[str] = Field(
        default=None, description="API key for WANDB, REQUIRED"
    )

    model_config = SettingsConfigDict(
        env_file=str(root_path / ".env"), yaml_file=[str(root_path / "config.yaml")]
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


@try_except(error_callable=exit)
def get_config() -> AppConfig:
    return AppConfig()


config = get_config()
