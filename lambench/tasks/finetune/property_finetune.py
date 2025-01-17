import os
from pathlib import Path
from typing import ClassVar

from lambench.models.basemodel import BaseLargeAtomModel
from lambench.tasks.base_task import BaseTask
from lambench.databases.property_table import PropertyRecord

from pydantic import BaseModel, ConfigDict


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
        self.get_property_json()
        return model.evaluate(self)

    def get_property_json(self):
        # Generate an input.json file
        # FIXME: needs to ensure workdir is created somewhere else, e.g. in dflow
        os.chdir(
            self.workdir
        )  # Needs to change here to ensure the model eval part is correct
        with open("input.json", "w") as _:
            # TODO: migrate from lamstare.utils.property.get_property_json
            raise NotImplementedError
