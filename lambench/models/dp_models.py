from pathlib import Path
from typing import Optional

from deepmd.main import main as deepmd_main

from lambench.tasks.base_task import BaseTask
from lambench.tasks.direct.direct_predict import DirectPredictTask
from lambench.tasks.finetune.property_finetune import PropertyFinetuneTask
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.tasks.utils import parse_dptest_log_file


class DPModel(BaseLargeAtomModel):
    # this need to be modified based on tasks
    DP_TASK_CONFIG: dict[str,tuple[str,bool]] = {
        # dataaset name: (head name, whether to change_bias)
        "ANI": ("Domains_Drug", False),
        # ...
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model_type != "DP":
            raise ValueError(
                f"Model type {self.model_type} is not supported by DPModel"
            )
        assert self.model_path is not None, "model_path should be specified"
    def evaluate(
        self, task: BaseTask
    ) -> Optional[dict[str, float]]:
        if isinstance(task, DirectPredictTask):
            if task.name not in self.DP_TASK_CONFIG:
                raise ValueError(f"Task {task.name} is not specified by DPModel")
            head, change_bias = self.DP_TASK_CONFIG[task.name]
        elif isinstance(task, PropertyFinetuneTask):
            head, change_bias = None, False
        else:
            raise ValueError(f"Task {task} is not supported by DPModel")
        self.model = self.model_path # Initialize the model
        assert task.workdir==Path.cwd(), "workdir should be the current working directory"

        if isinstance(task, PropertyFinetuneTask):
            self._finetune()
        elif change_bias:
            self._change_bias(task.test_data, head)
        self._freeze(head)
        self._test(task.test_data)
        result = parse_dptest_log_file(
            dataset_name=task.record_name, filepath=self.test_output
        )
        return result

    def _finetune(self):
        # Note: the input.json file is created under task.workdir
        command = f"dp --pt train input.json --finetune {self.model} --skip-neighbor-stat"
        deepmd_main(command.split()[1:])
        self.model = Path("model.ckpt.pt") # hard coded in deepmd-kit

    def _freeze(self, head=None):
        frozen_model = self.model.with_suffix(".pth")
        command = f"dp --pt freeze -c {self.model} -o {frozen_model} {f'--head {head}' if head else ''}"
        deepmd_main(command.split()[1:])
        self.model = frozen_model

    def _change_bias(self, test_file: Path, head:Optional[str]=None):
        change_bias_model = Path(f"change-bias-{self.model.name}")
        command = f"dp --pt change-bias {self.model.name} -o {change_bias_model} -f {test_file} {f'--model-branch {head}' if head else ''}"
        deepmd_main(command.split()[1:])
        self.model = change_bias_model

    def _test(self, test_file: Path):
        self.test_output = Path("dptest_output.txt")
        command = f"dp --pt test -m {self.model} -f {test_file} -l {self.test_output}"
        deepmd_main(command.split()[1:])
