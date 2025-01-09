from abc import abstractmethod
import tempfile
from pydantic import BaseModel, ConfigDict
from typing import Any
from pathlib import Path
class BaseTask(BaseModel):
    task_name: str
    test_data: Path
    model_config = ConfigDict(extra='allow')
    workdir: Path = Path(tempfile.gettempdir()) / "lambench"
    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def fetch_result(self):
        pass

    @abstractmethod
    def run_task(self, model):
        pass

    def prepare_test_data(self) -> Path:
        """
        This function should prepare a `test_data_{task_name}.txt` in the current working directory.
        """
        if not self.test_data or not self.test_data.exists():
            raise RuntimeError(f"Test data {self.test_data} does not exist.")

        temp_file_path = Path(f"{self.task_name}_test_file.txt")
        with temp_file_path.open('w') as f:
            for sys in self.test_data.rglob("type_map.raw"):
                f.write(f"{sys.parent}\n")
        return temp_file_path
