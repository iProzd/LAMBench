from lambench.models.basemodel import BaseLargeAtomModel
from pathlib import Path
import os
import logging
from lambench.tasks.utils import read_dptest_log_file
class DPModel(BaseLargeAtomModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model_type != "DP":
            raise ValueError(f"Model type {self.model_type} is not supported by DPModel")

    def evaluate(self, datapath: str, target_name: str, change_bias: bool = False, finetune: bool = False, head: str = None) -> dict:
        pass

    def _dp_test(self, ):
        pass
    
    def _freeze(self, head=None) -> None:
        pass

    def _change_bias(self, datapath: str, head=None) -> None:
        pass

    def _finetune(self):
        pass
