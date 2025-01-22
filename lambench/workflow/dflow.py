import logging
import os
from pathlib import Path
from types import NoneType

from dotenv import load_dotenv

import lambench
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.tasks.base_task import BaseTask
from lambench.workflow.entrypoint import job_list

load_dotenv(override=True)
# ruff: noqa: E402
from dflow import Task, Workflow
from dflow.plugins.bohrium import BohriumDatasetsArtifact, create_job_group
from dflow.plugins.dispatcher import DispatcherExecutor
from dflow.python import OP, Artifact, PythonOPTemplate

import deepmd


@OP.function
def run_task_op(
    task: BaseTask,
    model: BaseLargeAtomModel,
    dataset: Artifact(Path),  # type: ignore
) -> NoneType:
    task.run_task(model)


def submit_tasks_dflow(
    jobs: job_list,
    name="lambench",
    image="registry.dp.tech/dptech/dp/native/prod-375/lambench:v1",
    machine_type="1 * NVIDIA V100_32g",
):
    dataset_paths = [
        "/bohr/lambench-model-55c1/v3/",
        "/bohr/lambench-property-i0t1/v3/",
        "/bohr/lambench-ood-3z0s/v6/",
    ]
    job_group_id: int = create_job_group(name)
    logging.info(
        "Job group created: "
        f"https://www.bohrium.com/jobs/list?id={job_group_id}&groupName={name}&version=v2"
    )
    wf = Workflow(name=name)
    for task, model in jobs:
        dflow_task = Task(
            name=f"{task.task_name}_{model.model_name}".replace("_", "-"),
            template=PythonOPTemplate(
                run_task_op,  # type: ignore
                image=image,
                envs={k: v for k, v in os.environ.items() if k.startswith("MYSQL")},
                python_packages=[
                    Path(package.__path__[0]) for package in [lambench, deepmd]
                ],
            ),
            parameters={
                "task": task,
                "model": model,
            },
            artifacts={
                "dataset": [
                    BohriumDatasetsArtifact(dataset_path)
                    for dataset_path in dataset_paths
                ],
            },
            executor=DispatcherExecutor(
                machine_dict={
                    "batch_type": "Bohrium",
                    "context_type": "Bohrium",
                    "remote_profile": {
                        "input_data": {
                            "job_type": "container",
                            "bohr_job_group_id": job_group_id,
                            "platform": "ali",
                            "scass_type": machine_type,
                        },
                    },
                },
                resources_dict={
                    "source_list": [],  # for future use
                },
            ),
        )
        wf.add(dflow_task)
    wf_id = wf.submit()
    return wf_id
