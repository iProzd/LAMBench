import pytest
from unittest.mock import patch
from lambench.tasks.direct.direct_predict import DirectPredictTask
from lambench.databases.direct_predict_table import DirectPredictRecord

@pytest.fixture
def direct_yml_data():
    return {
        "Example_task": {
            "test_data": "oss://lambench/direct/Example_task/testdata/1",
            "energy_weight": 1.0,
            "force_weight": 1.0,
            "virial_weight": None
        }
    }

@pytest.fixture
def finetune_yml_data():
    return {
        "Example_task": {
            "property_name": "dipole_moment",
            "intensive": False,
            "property_dim": 1,
            "train_data": "oss://lambench/Example_task/train/1",
            "test_data": "oss://lambench/Example_task/test/1",
            "train_steps": 1000,
            "property_weight": 1.0
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
    return {
        "record_name": "model1#1000#taskA",
        "test_data":"test1",
        "energy_weight": 1.0,
        "force_weight": 2.0,
        "virial_weight": None
    }

@pytest.fixture
def finetune_task_data():
    return {
        "record_name": "model1#1000#taskA",
        "property_name": "dipole_moment",
        "intensive": False,
        "property_dim": 1,
        "train_data": "oss://lambench/Example_task/train/1",
        "test_data": "oss://lambench/Example_task/test/1",
        "train_steps": 1000,
        "property_weight": 1.0
    }
