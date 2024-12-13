from lambench.tasks.base_task import BaseTask
from lambench.databases.direct_predict_table import DirectPredictRecord
from typing import Dict, List 
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
        
    def run_task(self, model):
        test_data_file: str = self.prepare_test_data()
        task_output: Dict = model.evaluate(
            task_name=self.record_name.split("#")[1],
            test_file_path=test_data_file,
            target_name="standard",
        )
        return task_output

    def fetch_result(self) -> DirectPredictRecord:
        records = DirectPredictRecord.query_by_name(self.record_name)
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
            DirectPredictRecord(
                model_id=model_id,
                record_name=self.record_name,
                task_name=task_name,
                **task_output
            ).insert()
