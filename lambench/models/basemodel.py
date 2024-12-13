from pydantic import BaseModel
from typing import Dict
from enum import Enum
from abc import abstractmethod

class ModelType(str, Enum):
    DP = "DP"
    ASE = "ASE"

class BaseLargeAtomModel(BaseModel):
    model_id: str
    model_type: ModelType
    model_path: str
    virtualenv: str
    model_metadata: Dict[str, str]

    @abstractmethod
    def evaluate(self, data, target_name: str) -> Dict[str, float]:
        pass
