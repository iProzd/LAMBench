from lambench.tasks.base_task import BaseTask
from lambench.databases.property_table import PropertyRecord
from typing import Dict, List 
from typing import Optional
import logging

class PropertyFinetuneTask(BaseTask):
    """
    Support property finetuning and testing for DP interface.
    Currently does not support ASE interface.
    """

    property_name: str = None
    intensive: bool = True
    property_dim: int = 1
    train_data: str = None
    train_steps: int = 1000
    property_weight:float = None

    def __init__(self, record_name: str, **kwargs):
        super().__init__(record_name=record_name, test_data=kwargs['test_data'])
        self.property_name = kwargs.get("property_name", None)
        self.intensive = kwargs.get("intensive",None)
        self.property_dim = kwargs.get("property_dim", None) 
        self.train_data = kwargs.get("train_data", None) 
        self.train_steps = kwargs.get("train_steps", None) 
        self.property_weight = kwargs.get("property_weight", None)  
        
    def run_task(self, model):
        pass
        

    def fetch_result(self) -> PropertyRecord:
        records = PropertyRecord.query_by_name(self.record_name)
        if len(records) == 1:
            return records[0]
        elif len(records) > 1:
            logging.warning(f"Multiple records found for task {self.record_name}")
            return records[0]
        else:
            return None

    def sync_result(self, model) -> None:
        result = self.fetch_result()
        if result is not None:
            logging.info(f"TASK {self.record_name} record found in database, SKIPPING.")
            return
        else:
            task_output = self.run_task(model)
            logging.info(f"TASK {self.record_name} OUTPUT: {task_output}, INSERTING.")
            model_id, task_name = self.record_name.split("#")
            PropertyRecord(
                model_id=model_id,
                record_name=self.record_name,
                task_name=task_name,
                **task_output
            ).insert()