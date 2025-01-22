import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional, Type, TypeAlias

import yaml

import lambench
from lambench.models.ase_models import ASEModel
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.models.dp_models import DPModel
from lambench.tasks import DirectPredictTask, PropertyFinetuneTask
from lambench.tasks.base_task import BaseTask

DIRECT_TASKS = Path(lambench.__file__).parent / "tasks/direct/direct_tasks.yml"
FINETUNE_TASKS = Path(lambench.__file__).parent / "tasks/finetune/finetune_tasks.yml"
MODELS = Path(lambench.__file__).parent / "models/models_config.yml"


def gather_models(
    model_names: Optional[list[str]] = None,
) -> list[BaseLargeAtomModel]:
    """
    Gather models from the models_config.yml file.
    """

    models = []
    with open(MODELS, "r") as f:
        model_config: list[dict] = yaml.safe_load(f)
    for model_param in model_config:
        if model_names and model_param["model_name"] not in model_names:
            continue
        if model_param["model_type"] == "DP":
            models.append(DPModel(**model_param))
        elif model_param["model_type"] == "ASE":
            models.append(ASEModel(**model_param))
        else:
            raise ValueError(
                f"Model type {model_param['model_type']} is not supported."
            )
    return models


job_list: TypeAlias = list[tuple[BaseTask, BaseLargeAtomModel]]


def gather_task_type(
    models: list[BaseLargeAtomModel],
    task_file: Path,
    task_class: Type[BaseTask],
    task_names: Optional[list[str]] = None,
) -> job_list:
    """
    Gather tasks of a specific type from the task file.
    """
    tasks = []
    with open(task_file, "r") as f:
        task_configs: dict[str, dict] = yaml.safe_load(f)
    for model in models:
        if isinstance(model, ASEModel) and not issubclass(
            task_class, DirectPredictTask
        ):
            continue  # ASEModel only supports DirectPredictTask
        for task_name, task_params in task_configs.items():
            if task_names and task_name not in task_names:
                continue
            task = task_class(task_name=task_name, **task_params)
            if not task.exist(model.model_name):
                tasks.append((task, model))
    return tasks


def gather_jobs(
    model_names: Optional[list[str]] = None,
    task_names: Optional[list[str]] = None,
) -> job_list:
    jobs: job_list = []

    models = gather_models(model_names)
    if not models:
        logging.warning("No models found, skipping task gathering.")
        return jobs

    logging.info(f"Found {len(models)} models, gathering tasks.")
    jobs.extend(gather_task_type(models, DIRECT_TASKS, DirectPredictTask, task_names))
    jobs.extend(gather_task_type(models, FINETUNE_TASKS, PropertyFinetuneTask, task_names))
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run tasks for models.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="The model names in `models_config.yml`. e.g. --models DP_2024Q4 MACE_MP_0 SEVENNET_0",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        help="The task names in `direct_tasks.yml` or `finetune_tasks.yml`. e.g. --tasks HPt_NC_2022 Si_ZEO22",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run tasks locally.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    jobs = gather_jobs(model_names=args.models, task_names=args.tasks)
    if not jobs:
        logging.warning("No jobs found, exiting.")
        return
    logging.info(f"Found {len(jobs)} jobs.")
    if args.local:
        submit_tasks_local(jobs)
    else:
        from lambench.workflow.dflow import submit_tasks_dflow
        submit_tasks_dflow(jobs)


def submit_tasks_local(jobs: job_list) -> None:
    for task, model in jobs:
        logging.info(f"Running task={task.task_name}, model={model.model_name}")
        run_task(task, model)


def run_task(
    task: BaseTask,
    model: BaseLargeAtomModel,
) -> None:
    try:
        task.run_task(model)
    except ModuleNotFoundError as e:
        logging.error(e)  # Import error for ASE models
    except Exception as _:
        traceback.print_exc()
        logging.error(f"task={task.task_name}, model={model.model_name} failed!")


if __name__ == "__main__":
    main()
