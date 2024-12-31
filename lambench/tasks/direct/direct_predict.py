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

    energy_weight: Optional[float] = None
    force_weight: Optional[float] = None
    virial_weight: Optional[float] = None

    def __init__(self, record_name: str, **kwargs):
        super().__init__(record_name=record_name, test_data=kwargs['test_data'])
        self.energy_weight = kwargs.get("energy_weight", None)
        self.force_weight = kwargs.get("force_weight",None)
        self.virial_weight = kwargs.get("virial_weight", None)
        self.name=self.record_name.split("#")[1],
        self.test_file_path=self.prepare_test_data(),
        self.target_name="standard",

    def evaluate(self, model: BaseLargeAtomModel):
        task_output: dict = model.evaluate(self)
        return task_output

    def fetch_result(self) -> Optional[DirectPredictRecord]:
        records = DirectPredictRecord.query_by_name(self.record_name)
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
            DirectPredictRecord(
                model_id=model_id,
                record_name=self.record_name,
                task_name=task_name,
                **task_output
            ).insert()
