from abc import abstractmethod
from pydantic import BaseModel, ConfigDict
from lambench.databases.base_table import BaseRecord
from typing import List, Dict, Any
from pydantic import validator
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
