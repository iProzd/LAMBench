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

    @abstractmethod
    def evaluate(self, task) -> dict[str, float]:
        pass
