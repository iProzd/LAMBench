from lambench.workflow.entrypoint import MODELS, DIRECT_TASKS, FINETUNE_TASKS
import yaml
from lambench.models.ase_models import ASEModel
from lambench.models.dp_models import DPModel
from lambench.tasks.base_task import BaseTask


def test_validate_config():
    assert MODELS.exists(), f"MODELS file {MODELS} does not exist."
    assert DIRECT_TASKS.exists(), f"DIRECT_TASKS file {DIRECT_TASKS} does not exist."
    assert (
        FINETUNE_TASKS.exists()
    ), f"FINETUNE_TASKS file {FINETUNE_TASKS} does not exist."

    model_config_error_list = []
    direct_task_config_error_list = []
    finetune_task_config_error_list = []

    with open(MODELS, "r") as f:
        model_config = yaml.safe_load(f)

    for model_param in model_config:
        if model_param["model_type"] == "DP":
            clstype = DPModel
        elif model_param["model_type"] == "ASE":
            clstype = ASEModel
        else:
            model_config_error_list.append(
                f"Model type {model_param['model_type']} is not supported."
            )
            continue
        try:
            clstype(**model_param)
        except Exception as e:
            model_config_error_list.append(f"Error in model config {model_param}: {e}")

    for task_file in [DIRECT_TASKS, FINETUNE_TASKS]:
        with open(task_file, "r") as f:
            task_configs = yaml.safe_load(f)
        for task_name, task_param in task_configs.items():
            try:
                BaseTask(task_name=task_name, **task_param)
            except Exception as e:
                if task_file == DIRECT_TASKS:
                    direct_task_config_error_list.append(
                        f"Error in direct task config {task_param}: {e}"
                    )
                else:
                    finetune_task_config_error_list.append(
                        f"Error in finetune task config {task_param}: {e}"
                    )

    assert (
        not model_config_error_list
    ), f"Model config errors: {model_config_error_list}"
    assert (
        not direct_task_config_error_list
    ), f"Direct task config errors: {direct_task_config_error_list}"
    assert (
        not finetune_task_config_error_list
    ), f"Finetune task config errors: {finetune_task_config_error_list}"
