import pytest
from unittest.mock import patch
from lambench.tasks.direct.direct_predict import DirectPredictTask
from lambench.databases.direct_predict_table import DirectPredictRecord
@pytest.fixture
def direct_predict_task_data():
    return {
        "Example_task": {
            "test_data": [
                "oss://lambench/direct/Example_task/testdata/1",
                "oss://lambench/direct/Example_task/testdata/2"
            ],
            "energy_weight": 1.0,
            "force_weight": 1.0,
            "virial_weight": None
        }
    }



@pytest.fixture
def mock_direct_predict_record():
    with patch("lambench.tasks.direct.direct_predict.DirectPredictRecord") as mock_record:
        yield mock_record


@pytest.fixture
def task_data():
    return {
        "record_name": "model1#1000#taskA",
        "test_data": ["test1", "test2"],
        "energy_weight": 1.0,
        "force_weight": 2.0,
        "virial_weight": None
    }