from pydantic import BaseModel
from typing import Dict, Optional
from enum import Enum
from abc import abstractmethod
class ModelType(str, Enum):
    DP = "DP"
    ASE = "ASE"

class BaseLargeAtomModel(BaseModel):
    model_id: str
    model_type: ModelType
    model_path: Optional[str]
    virtualenv: str
    model_metadata: Dict[str, str]

    @abstractmethod
    def evaluate(self, task_name:str, test_file_path: str, target_name: str) -> Dict[str, float]:
        pass
