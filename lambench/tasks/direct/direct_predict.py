from lambench.tasks.base_task import BaseTask
from lambench.databases.direct_predict_table import DirectPredictRecord
from typing import Dict, List 
from typing import Optional

class DirectPredictTask(BaseTask):
    energy_weight: Optional[float] = None
    force_weight: Optional[float] = None
    virial_weight: Optional[float] = None  
    result: Optional[DirectPredictRecord] = None 

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, test_data=kwargs['test_data'])
        self.energy_weight = kwargs.get("energy_weight", None)
        self.force_weight = kwargs.get("force_weight",None)
        self.virial_weight = kwargs.get("virial_weight", None)  
        self.result = None
        
    def run_task(self):
        self.result = self.database.predict(self.test_data)

    def fetch_result(self) -> DirectPredictRecord:
        return self.result

    def sync_result(self):
        pass

    def show_result(self):
        pass