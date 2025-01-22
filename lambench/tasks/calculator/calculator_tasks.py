from typing import ClassVar
from lambench.models.ase_models import ASEModel
from lambench.tasks.base_task import BaseTask
from lambench.databases.calculator_table import CalculatorRecord


class CalculatorTask(BaseTask):
    """
    Support more general calculator tasks interfaced with ASE.
    """

    record_type: ClassVar = CalculatorRecord

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, test_data=kwargs["test_data"])

    def evaluate(self, model: ASEModel) -> dict[str, float]:
        """
        Evaluate the task for the model.
        """
        if self.task_name == "nve_md":
            from lambench.tasks.calculator.nve_md import run_md_nve_simulation

            return run_md_nve_simulation(model)
        else:
            pass
