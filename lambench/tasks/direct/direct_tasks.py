from typing import ClassVar
from lambench.tasks.base_task import BaseTask
from lambench.databases.direct_predict_table import DirectPredictRecord


class DirectPredictTask(BaseTask):
    """
    Support direct energy force prediction for DP interface, and zero-shot energy force prediciton for DP interface.
    For models using the ASE interface, should use `DirectPredictASETask` instead.
    """

    record_type: ClassVar = DirectPredictRecord

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, test_data=kwargs["test_data"])
        # self.test_file_path = self.prepare_test_data()
