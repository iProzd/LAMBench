import os
from pathlib import Path
from typing import ClassVar

from lambench.models.basemodel import BaseLargeAtomModel
from lambench.tasks.base_task import BaseTask
from lambench.databases.property_table import PropertyRecord


class PropertyFinetuneTask(BaseTask):
    """
    Support property finetuning and testing for DP interface.
    Currently does not support ASE interface.
    """

    record_type: ClassVar = PropertyRecord
    property_name: str
    intensive: bool = True
    property_dim: int = 1
    train_data: Path
    train_steps: int = 1000

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, **kwargs, target_name="finetune")

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
