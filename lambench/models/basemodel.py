from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from enum import Enum
from abc import abstractmethod


class ModelType(str, Enum):
    DP = "DP"
    ASE = "ASE"


class SkipTaskType(str, Enum):
    DirectPredictTask = "DirectPredictTask"
    PropertyFinetuneTask = "PropertyFinetuneTask"
    CalculatorTask = "CalculatorTask"


class BaseLargeAtomModel(BaseModel):
    model_name: str
    model_family: str
    model_type: ModelType
    model_path: Optional[Path] = None
    virtualenv: str
    model_metadata: dict[str, str]
    show_direct_task: bool = True
    show_finetune_task: bool = False
    show_calculator_task: bool = False
    skip_tasks: list[SkipTaskType] = []

    @abstractmethod
    def evaluate(self, task) -> dict[str, float]:
        pass
