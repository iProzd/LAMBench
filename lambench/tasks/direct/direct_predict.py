from lambench.tasks.base_task import BaseTask

class DirectPredictTask(BaseTask):
    def __init__(self, database, test_data):
        super().__init__(database, test_data)
        self.result = None

    def run_task(self, task_params=None):
        self.result = self.database.predict(self.test_data)

    def fetch_result(self, task_params=None):
        return self.result

    def sync_result(self, task_params=None):
        pass

    def show_result(self):
        pass