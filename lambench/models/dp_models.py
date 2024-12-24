from lambench.models.basemodel import BaseLargeAtomModel
from pathlib import Path
import os
import logging
from typing import Dict
from lambench.tasks.utils import read_dptest_log_file
class DPModel(BaseLargeAtomModel):

    #this need to be modified based on tasks
    DP_TASK_CONFIG: Dict = {
        "Example_task_A" :("head_A", True), # head, change_bias
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model_type != "DP":
            raise ValueError(f"Model type {self.model_type} is not supported by DPModel")

    def evaluate(self, task_name:str, test_file_path: str, target_name: str) -> dict:
        if target_name != "standard":
            self._finetune()
        if task_name not in self.DP_TASK_CONFIG:
            logging.error(f"Task {task_name} is not specified by DPModel")
            return {}
        head, change_bias = self.DP_TASK_CONFIG[task_name]
        if change_bias:
            self._change_bias(test_file_path, head)
        else:
            self._freeze(head)
        self._dp_test()
        result = read_dptest_log_file("dptest.log")
        return result

    def _dp_test(self, ):
        pass
    
    def _freeze(self, head=None) -> None:
        pass

    def _change_bias(self, test_file_path: str, head=None) -> None:
        pass

    def _finetune(self):
        pass
