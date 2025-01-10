import logging
import os
from pathlib import Path
from typing import Optional

from deepmd.main import main as deepmd_main

from lambench.tasks.direct.direct_predict import DirectPredictTask
from lambench.tasks.finetune.property_finetune import PropertyFinetuneTask
from lambench.tasks.utils import parse_dptest_log_file
from lambench.tasks.base_task import BaseTask
from lambench.models.basemodel import BaseLargeAtomModel


class DPModel(BaseLargeAtomModel):
    # this need to be modified based on tasks
    DP_TASK_CONFIG: dict[str, tuple[str, bool]] = {
        # dataaset name: (head name, whether to change_bias)
        "ANI": ("Domains_Drug", True),
        # ...
    }
    model_path: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model_type != "DP":
            raise ValueError(
                f"Model type {self.model_type} is not supported by DPModel"
            )

    def evaluate(self, task: BaseTask) -> Optional[dict[str, float]]:
        if isinstance(task, DirectPredictTask):
            if task.task_name not in self.DP_TASK_CONFIG:
                raise ValueError(f"Task {task.task_name} is not specified by DPModel")
            head, change_bias = self.DP_TASK_CONFIG[task.task_name]
        elif isinstance(task, PropertyFinetuneTask):
            head, change_bias = None, False
        else:
            raise ValueError(f"Task {task} is not supported by DPModel")
        task.workdir.mkdir(exist_ok=True)
        os.chdir(task.workdir)

        model = self.model_path
        if isinstance(task, PropertyFinetuneTask):
            model = self._finetune(model)
        elif change_bias:
            model = self._change_bias(model, task.test_data, head)
        model = self._freeze(model, head)
        test_output = self._test(model, task.test_data, head)
        result = parse_dptest_log_file(filepath=test_output)
        return result

    def _finetune(self, model: Path):
        # Note: the input.json file is created under task.workdir
        command = f"dp --pt train input.json --finetune {model} --skip-neighbor-stat"
        deepmd_main(command.split()[1:])
        return Path("model.ckpt.pt")  # hard coded in deepmd-kit

    def _freeze(self, model: Path, head=None):
        frozen_model = Path.cwd() / model.with_suffix(".pth").name
        command = f"dp --pt freeze -c {model} -o {frozen_model} {f'--head {head}' if head else ''}"
        deepmd_main(command.split()[1:])
        return frozen_model

    def _change_bias(self, model: Path, test_data: Path, head: Optional[str] = None):
        change_bias_model = Path.cwd() / f"change-bias-{model.name}"
        command = f"dp --pt change-bias {model} -o {change_bias_model} -s {test_data} {f'--model-branch {head}' if head else ''}"
        deepmd_main(command.split()[1:])
        return change_bias_model

    def _test(self, model:Path, test_data: Path, head: Optional[str] = None):
        test_output = Path("dptest_output.txt")
        command = f"dp --pt test -m {model} -s {test_data} -l {test_output} {f'--head {head}' if head else ''}"
        deepmd_main(command.split()[1:])
        return test_output
