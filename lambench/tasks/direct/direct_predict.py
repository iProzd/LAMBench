from lambench.models.basemodel import BaseLargeAtomModel
from lambench.tasks.base_task import BaseTask
from lambench.databases.direct_predict_table import DirectPredictRecord
from typing import Optional
import logging


class DirectPredictTask(BaseTask):
    """
    Support direct energy force prediction for DP interface, and zero-shot energy force prediciton for DP interface.
    For models using the ASE interface, should use `DirectPredictASETask` instead.
    """

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, test_data=kwargs["test_data"])
        self.test_file_path = self.prepare_test_data()
        self.target_name = "standard"

    def evaluate(self, model: BaseLargeAtomModel):
        task_output: dict = model.evaluate(self)
        return task_output

    def fetch_result(self) -> Optional[DirectPredictRecord]:
        records = DirectPredictRecord.query_by_name(self.task_name) # FIXME: by model name and task name
        if len(records) == 1:
            return records[0]
        elif len(records) > 1:
            logging.warning(f"Multiple records found for task {self.task_name}")
            return records[0]
        else:
            return None

    def run_task(self, model: BaseLargeAtomModel) -> None:
        result = self.fetch_result()
        if result is not None:
            logging.info(f"TASK {self.task_name} record found in database, SKIPPING.")
            return
        else:
            task_output = self.evaluate(model)
            logging.info(f"TASK {self.task_name} OUTPUT: {task_output}, INSERTING.")
            DirectPredictRecord(
                task_name=self.task_name,
                model_name=model.model_name,
                **task_output,
            ).insert()
