from typing import ClassVar, Any
from lambench.models.ase_models import ASEModel
from lambench.tasks.base_task import BaseTask
from lambench.databases.calculator_table import CalculatorRecord


class CalculatorTask(BaseTask):
    """
    Support more general calculator tasks interfaced with ASE.
    """

    record_type: ClassVar = CalculatorRecord
    calculator_params: dict[str, Any]

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, test_data=kwargs["test_data"])

    def run_task(self, model: ASEModel) -> dict[str, float]:
        """
        Evaluate the task for the model.
        """
        if self.task_name == "nve_md":
            from lambench.tasks.calculator.nve_md import run_md_nve_simulation

            num_steps = self.calculator_params.get("num_steps", 1000)
            timestep = self.calculator_params.get("timestep", 1.0)
            temperature_K = self.calculator_params.get("temperature_K", 300)
            return run_md_nve_simulation(model, num_steps, timestep, temperature_K)
        else:
            pass
