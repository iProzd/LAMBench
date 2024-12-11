from abc import abstractmethod
from pydantic import BaseModel
from lambench.databases.base_table import BaseRecord
from typing import List, Dict
from pydantic import validator
class BaseTask(BaseModel):
    task_name: str
    test_data: List[str]

    @validator('test_data')
    def validate_test_data(cls, v):
        if not isinstance(v, list) or not all(isinstance(item, str) for item in v):
            raise ValueError('"testdata" must be a list of strings') 
        return v

    class Config:
        arbitrary_types_allowed=True
        
    @abstractmethod
    def run_task(self):
        pass

    @abstractmethod
    def fetch_result(self):
        pass

    @abstractmethod
    def sync_result(self):
        pass

    @abstractmethod
    def show_result(self):
        pass