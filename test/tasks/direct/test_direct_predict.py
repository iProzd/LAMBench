from lambench.tasks import DirectPredictTask
from lambench.models.dp_models import DPModel
from unittest.mock import MagicMock, patch
import logging

def test_load_direct_predict_task(direct_yml_data):
    for task_name, task_param in direct_yml_data.items():
        model_id = "TEST_DP_v1"
        record_name = f"{model_id}#{task_name}"
        task = DirectPredictTask(record_name=record_name,**task_param)
        for key, value in task_param.items():
            assert getattr(task, key) == value


def test_fetch_result_single_record(mock_direct_predict_record, direct_task_data):
    mock_record_instance = MagicMock()
    mock_direct_predict_record.query_by_name.return_value = [mock_record_instance]

    task = DirectPredictTask(**direct_task_data)
    result = task.fetch_result()
    mock_direct_predict_record.query_by_name.assert_called_once_with(direct_task_data["record_name"])
    assert result == mock_record_instance


def test_fetch_result_multiple_records(mock_direct_predict_record, direct_task_data, caplog):
    record1 = MagicMock()
    record2 = MagicMock()
    mock_direct_predict_record.query_by_name.return_value = [record1, record2]

    task = DirectPredictTask(**direct_task_data)
    with caplog.at_level(logging.WARNING):
        result = task.fetch_result()

    mock_direct_predict_record.query_by_name.assert_called_once_with(direct_task_data["record_name"])
    assert result == record1
    assert f"Multiple records found for task {direct_task_data['record_name']}" in caplog.text

def test_fetch_result_no_records(mock_direct_predict_record, direct_task_data):
    mock_direct_predict_record.query_by_name.return_value = []

    task = DirectPredictTask(**direct_task_data)
    result = task.fetch_result()

    mock_direct_predict_record.query_by_name.assert_called_once_with(direct_task_data["record_name"])
    assert result is None

def test_run_task_existing_record(mock_direct_predict_record, valid_model_data, direct_task_data, caplog):
    existing_record = MagicMock()
    mock_direct_predict_record.query_by_name.return_value = [existing_record]

    task = DirectPredictTask(**direct_task_data)
    model = DPModel(**valid_model_data)
    with caplog.at_level(logging.INFO):
        task.run_task(model)

    assert f"TASK {direct_task_data['record_name']} record found in database, SKIPPING." in caplog.text
    mock_direct_predict_record.query_by_name.assert_called_once_with(direct_task_data["record_name"])


def test_run_task_no_existing_record(mock_direct_predict_record, valid_model_data, direct_task_data, caplog):
    mock_direct_predict_record.query_by_name.return_value = []

    # Mock the evaluate method
    model = DPModel(**valid_model_data)
    with patch.object(DirectPredictTask, "evaluate", return_value={"some_field": "some_value"}) as mock_evaluate, caplog.at_level(logging.INFO):
        task = DirectPredictTask(**direct_task_data)
        task.run_task(model)

    mock_evaluate.assert_called_once()
    assert f"TASK {direct_task_data['record_name']} OUTPUT: {{'some_field': 'some_value'}}, INSERTING." in caplog.text
    mock_direct_predict_record.query_by_name.assert_called_once_with(direct_task_data["record_name"])
    mock_direct_predict_record.assert_called_with(
        model_id="model1",
        record_name=direct_task_data["record_name"],
        task_name="taskA",
        some_field="some_value"
    )