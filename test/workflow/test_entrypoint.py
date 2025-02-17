from lambench.models.dp_models import DPModel
from lambench.tasks import PropertyFinetuneTask
import pytest
from lambench.workflow.entrypoint import gather_task_type


def _create_dp_model(skip_tasks=[]):
    return DPModel(
        model_name="test_model",
        model_family="test_family",
        model_type="DP",
        model_path="test_path",
        virtualenv="test_env",
        model_metadata={"test": "test"},
        skip_tasks=skip_tasks,
    )


@pytest.fixture
def dp_model():
    return _create_dp_model()


@pytest.fixture
def dp_model_skip_tasks():
    return _create_dp_model(skip_tasks=["PropertyFinetuneTask"])


def test_gather_task_type_with_skip(dp_model, dp_model_skip_tasks):
    models = [dp_model, dp_model_skip_tasks]
    task_class = PropertyFinetuneTask
    tasks = gather_task_type(models, task_class)
    assert len(tasks) == 40
