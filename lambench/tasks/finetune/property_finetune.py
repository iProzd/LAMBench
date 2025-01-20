import os
from pathlib import Path
from typing import ClassVar

from lambench.models.basemodel import BaseLargeAtomModel
from lambench.tasks.base_task import BaseTask
from lambench.databases.property_table import PropertyRecord

from pydantic import BaseModel, ConfigDict
import copy
import json


class FinetuneParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = 512
    ngpus: int = 1
    start_lr: float = 1e-3
    stop_lr: float = 1e-4
    train_steps: int = 100000


class PropertyFinetuneTask(BaseTask):
    """
    Support property finetuning and testing for DP interface.
    Currently does not support ASE interface.
    """

    record_type: ClassVar = PropertyRecord
    property_name: str  # The name of the property to be finetuned, e.g. dielectric
    intensive: bool = True
    property_dim: int = 1
    train_data: Path
    finetune_params: FinetuneParams

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, **kwargs)

    def evaluate(self, model: BaseLargeAtomModel):
        return model.evaluate(self)

    def prepare_property_directory(self, model: BaseLargeAtomModel):
        assert (
            os.getcwd() == self.workdir
        ), f"Current working directory is {os.getcwd()}, need to change working directory to {self.workdir}!"

        # 1. write the finetune input.json file
        with open(os.path.join(model.model_path.parent, "input.json"), "r") as f:
            pretrain_config = json.load(f)

        finetune_config = copy.deepcopy(pretrain_config)

        # 2. modify the input.json file
        finetune_config["learning_rate"] = {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": self.finetune_params.get("start_lr", 1e-3),
            "stop_lr": self.finetune_params.get("stop_lr", 1e-4),
            "_comment": "that's all",
        }

        finetune_config["model"]["fitting_net"] = {
            "type": "property",
            "property_name": self.property_name,
            "intensive": self.intensive,
            "task_dim": self.property_dim,
            "neuron": [240, 240, 240],
            "resnet_dt": True,
            "seed": 1,
            "_comment": " that's all",
        }

        finetune_config["loss"] = {"type": "property", "_comment": " that's all"}

        finetune_config["training"] = {
            "training_data": {
                "systems": str(self.train_data),
                "batch_size": self.finetune_params.get("batch_size", 512),
                "_comment": "that's all",
            },
            "validation_data": {
                "systems": str(self.test_data),
                "batch_size": 1,
                "_comment": "that's all",
            },
            "warmup_steps": 0,
            "gradient_max_norm": 5.0,
            "max_ckpt_keep": 10,
            "seed": 1,
            "disp_file": "lcurve.out",
            "disp_freq": self.finetune_params.get("train_steps", 100000) // 20,
            "numb_steps": self.finetune_params.get("train_steps", 100000),
            "save_freq": self.finetune_params.get("train_steps", 100000) // 5,
            "_comment": "that's all",
        }

        with open(os.path.join(self.workdir, "input.json"), "w") as f:
            json.dump(finetune_config, f, indent=4)
