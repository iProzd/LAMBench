import logging
import tempfile
from pydantic import BaseModel, ConfigDict
from pathlib import Path

from lambench.databases.base_table import BaseRecord
from lambench.models.basemodel import BaseLargeAtomModel
class BaseTask(BaseModel):
    task_name: str
    test_data: Path
    model_config = ConfigDict(extra='allow')
    workdir: Path = Path(tempfile.gettempdir()) / "lambench"
    record_type: type[BaseRecord] = BaseRecord

    def evaluate(self, model: BaseLargeAtomModel):
        task_output: dict = model.evaluate(self)
        return task_output

    def exist(self, model_name: str) -> bool:
        num_records = self.record_type.count(
            task_name=self.task_name, model_name=model_name
        )
        if num_records > 1:
            logging.warning(f"Multiple records found for task {self.task_name}")
        if num_records >= 1:
            return True
        else:
            return False

    def run_task(self, model: BaseLargeAtomModel) -> None:
        if self.exist(model.model_name):
            logging.info(f"TASK {self.task_name} record found in database, SKIPPING.")
            return
        else:
            task_output = self.evaluate(model)
            logging.info(f"TASK {self.task_name} OUTPUT: {task_output}, INSERTING.")
            self.record_type(
                task_name=self.task_name,
                model_name=model.model_name,
                **task_output,
            ).insert()

    def prepare_test_data(self) -> Path:
        """
        This function should prepare a `test_data_{task_name}.txt` in the current working directory.
        """
        if not self.test_data or not self.test_data.exists():
            raise RuntimeError(f"Test data {self.test_data} does not exist.")

        temp_file_path = Path(f"{self.task_name}_test_file.txt")
        with temp_file_path.open('w') as f:
            for sys in self.test_data.rglob("type_map.raw"):
                f.write(f"{sys.parent}\n")
        return temp_file_path
