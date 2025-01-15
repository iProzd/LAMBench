from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from enum import Enum
from abc import abstractmethod


class ModelType(str, Enum):
    DP = "DP"
    ASE = "ASE"


class BaseLargeAtomModel(BaseModel):
    model_name: str
    model_type: ModelType
    model_path: Optional[Path]
    virtualenv: str
    model_metadata: dict[str, str]
    show_direct_task: Optional[bool] = False
    show_finetune_task: Optional[bool] = False
    show_calculator_task: Optional[bool] = False

    @abstractmethod
    def evaluate(self, task) -> dict[str, float]:
        pass
