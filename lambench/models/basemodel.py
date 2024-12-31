from pathlib import Path
from pydantic import BaseModel
from enum import Enum
from abc import abstractmethod

from LAMBench.lambench.tasks.base_task import BaseTask
class ModelType(str, Enum):
    DP = "DP"
    ASE = "ASE"

class BaseLargeAtomModel(BaseModel):
    model_id: str
    model_type: ModelType
    model_path: Path
    virtualenv: str
    model_metadata: dict[str, str]

    @abstractmethod
    def evaluate(self, task: BaseTask) -> dict[str, float]:
        pass
