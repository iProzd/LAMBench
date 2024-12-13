from abc import abstractmethod
from pydantic import BaseModel, ConfigDict
from lambench.databases.base_table import BaseRecord
from typing import List, Dict, Any
from pydantic import validator
from pathlib import Path
import logging
class BaseTask(BaseModel):
    record_name: str
    test_data: str
    model_config = ConfigDict(extra='allow')
        
    @abstractmethod
    def run_task(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def fetch_result(self):
        pass

    @abstractmethod
    def sync_result(self, model):
        pass

    @classmethod
    def prepare_test_data(self):
        """
        This function should prepare a `test_data_{task_name}.txt` in the current working directory.
        """
        if not self.test_data or not Path(self.test_data).exists():
            logging.error(f"Test data {self.test_data} does not exist.")
            return None
        
        temp_file_path = Path(f"{self.record_name}_test_file.txt")
        with temp_file_path.open('w') as f:
            for sys in Path(self.test_data).rglob("type_map.raw"):
                f.write(f"{sys.parent}\n")
        return str(temp_file_path)