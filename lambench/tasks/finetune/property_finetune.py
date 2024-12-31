import os
from LAMBench.lambench.models.basemodel import BaseLargeAtomModel
from lambench.tasks.base_task import BaseTask
from lambench.databases.property_table import PropertyRecord
from typing import Optional
import logging

class PropertyFinetuneTask(BaseTask):
    """
    Support property finetuning and testing for DP interface.
    Currently does not support ASE interface.
    """

    property_name: str
    intensive: bool = True
    property_dim: int = 1
    train_data: str
    train_steps: int = 1000
    property_weight:float

    def __init__(self, record_name: str, **kwargs):
        super().__init__(record_name=record_name, test_data=kwargs['test_data'])
        self.property_name = kwargs.get("property_name", None)
        self.intensive = kwargs.get("intensive",None)
        self.property_dim = kwargs.get("property_dim", None)
        self.train_data = kwargs.get("train_data", None)
        self.train_steps = kwargs.get("train_steps", None)
        self.property_weight = kwargs.get("property_weight", None)

    def evaluate(self, model: BaseLargeAtomModel):
        self.get_property_json()
        return model.evaluate(self)

    def get_property_json(self):
        # Generate an input.json file
        # FIXME: needs to ensure workdir is created somewhere else, e.g. in dflow
        os.chdir(self.workdir) # Needs to change here to ensure the model eval part is correct
        with open("input.json", "w") as _:
            # TODO: migrate from lamstare.utils.property.get_property_json
            raise NotImplementedError

    def fetch_result(self) -> Optional[PropertyRecord]:
        records = PropertyRecord.query_by_name(self.record_name)
        if len(records) == 1:
            return records[0]
        elif len(records) > 1:
            logging.warning(f"Multiple records found for task {self.record_name}")
            return records[0]
        else:
            return None

    def run_task(self, model) -> None:
        result = self.fetch_result()
        if result is not None:
            logging.info(f"TASK {self.record_name} record found in database, SKIPPING.")
            return
        else:
            task_output = self.evaluate(model)
            logging.info(f"TASK {self.record_name} OUTPUT: {task_output}, INSERTING.")
            model_id, task_name = self.record_name.split("#")
            PropertyRecord(
                model_id=model_id,
                record_name=self.record_name,
                task_name=task_name,
                **task_output
            ).insert()
            # TODO: return the working dir to dflow containing trained mode
