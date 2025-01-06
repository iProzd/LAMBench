from pathlib import Path
import pytest
from unittest.mock import patch


def make_dummy_data():
    Path("data/dummy/test/1").mkdir(parents=True, exist_ok=True)
    Path("data/dummy/train/1").mkdir(parents=True, exist_ok=True)

# Fixtures for TASKS
@pytest.fixture
def direct_yml_data():
    make_dummy_data()
    return {
        "Example_task": {
            "test_data": Path("data/dummy/test/1"),
            "energy_weight": 1.0,
            "force_weight": 1.0,
            "virial_weight": None
        }
    }

@pytest.fixture
def finetune_yml_data():
    make_dummy_data()
    return {
        "Example_task": {
            "property_name": "dipole_moment",
            "intensive": False,
            "property_dim": 1,
            "train_data": Path("data/dummy/train/1"),
            "test_data": Path("data/dummy/test/1"),
            "train_steps": 1000,
            "property_weight": 1.0,
        }
    }

@pytest.fixture
def mock_direct_predict_record():
    with patch("lambench.tasks.direct.direct_predict.DirectPredictRecord") as mock_record:
        yield mock_record

@pytest.fixture
def mock_finetune_record():
    with patch("lambench.tasks.finetune.property_finetune.PropertyRecord") as mock_record:
        yield mock_record

@pytest.fixture
def direct_task_data():
    make_dummy_data()
    return {
        "record_name": "model1#taskA",
        "test_data": Path("data/dummy/test/1"),
        "energy_weight": 1.0,
        "force_weight": 2.0,
        "virial_weight": None,
    }

@pytest.fixture
def finetune_task_data():
    make_dummy_data()
    return {
        "record_name": "model1#taskA",
        "property_name": "dipole_moment",
        "intensive": False,
        "property_dim": 1,
        "train_data": Path("data/dummy/train/1"),
        "test_data": Path("data/dummy/test/1"),
        "train_steps": 1000,
        "property_weight": 1.0,
    }


# Fixtures for MODELS
@pytest.fixture
def valid_model_data():
    return {
        "model_id": "model1",
        "model_type": "DP",
        "model_path": "oss://lambench/DP/model.ckpt-1000.pt",
        "virtualenv": "oss://lambench/DP/model1/venv",
        "model_metadata": {
            "author": "author1",
            "description": "description1"
        }
    }

@pytest.fixture
def invalid_model_data():
    return {
        "model_id": "model1",
        "model_type": "Unknown",
        "model_path": None,
        "model_metadata": {
            "author": "author1",
            "description": "description1"
        }
    }