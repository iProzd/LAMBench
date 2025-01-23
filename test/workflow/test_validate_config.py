from lambench.workflow.entrypoint import (
    MODELS,
    DIRECT_TASKS,
    FINETUNE_TASKS,
    CALCULATOR_TASKS,
)
import yaml
from lambench.models.ase_models import ASEModel
from lambench.models.dp_models import DPModel
from lambench.tasks import DirectPredictTask, CalculatorTask, PropertyFinetuneTask


def test_validate_config():
    assert MODELS.exists(), f"MODELS file {MODELS} does not exist."
    assert DIRECT_TASKS.exists(), f"DIRECT_TASKS file {DIRECT_TASKS} does not exist."
    assert (
        FINETUNE_TASKS.exists()
    ), f"FINETUNE_TASKS file {FINETUNE_TASKS} does not exist."

    model_config_error_list = []
    direct_task_config_error_list = []
    finetune_task_config_error_list = []
    calculator_task_config_error_list = []

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

    task_files = {
        DIRECT_TASKS: (DirectPredictTask, direct_task_config_error_list),
        FINETUNE_TASKS: (PropertyFinetuneTask, finetune_task_config_error_list),
        CALCULATOR_TASKS: (CalculatorTask, calculator_task_config_error_list),
    }

    for task_file, (clstype, error_list) in task_files.items():
        with open(task_file, "r") as f:
            task_configs = yaml.safe_load(f)
        for task_name, task_param in task_configs.items():
            try:
                clstype(task_name=task_name, **task_param)
            except Exception as e:
                error_list.append(
                    f"Error in {task_file.stem} task config {task_param}: {e}"
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
    assert (
        not calculator_task_config_error_list
    ), f"Calculator task config errors: {calculator_task_config_error_list}"
