from abc import ABC, abstractmethod
from pydantic import BaseModel
from lambench.databases.base_table import BaseRecord
from typing import List
class BaseTask(BaseModel, ABC):
    database: BaseRecord
    test_data: List[str]

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