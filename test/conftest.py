import pytest

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