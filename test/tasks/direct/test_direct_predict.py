from lambench.tasks.direct.direct_predict import DirectPredictTask
import yaml
from unittest.mock import MagicMock, patch
import logging

def test_load_direct_predict_task(direct_predict_task_data):
    for task_name, task_param in direct_predict_task_data.items():
        model_id, step = "TEST_DP_v1", 1000
        record_name = f"{model_id}#{step}#{task_name}"
        task = DirectPredictTask(record_name=record_name,**task_param)
        assert task.test_data is not None
        assert task.energy_weight == task_param["energy_weight"]
        assert task.force_weight == task_param["force_weight"]
        assert task.virial_weight == task_param.get("virial_weight")


def test_fetch_result_single_record(mock_direct_predict_record, task_data):
    mock_record_instance = MagicMock()
    mock_direct_predict_record.query_by_name.return_value = [mock_record_instance]
    
    task = DirectPredictTask(**task_data)
    result = task.fetch_result()
    mock_direct_predict_record.query_by_name.assert_called_once_with(task_data["record_name"])
    assert result == mock_record_instance


def test_fetch_result_multiple_records(mock_direct_predict_record, task_data, caplog):
    record1 = MagicMock()
    record2 = MagicMock()
    mock_direct_predict_record.query_by_name.return_value = [record1, record2]
    
    task = DirectPredictTask(**task_data)
    with caplog.at_level(logging.WARNING):
        result = task.fetch_result()
    
    mock_direct_predict_record.query_by_name.assert_called_once_with(task_data["record_name"])
    assert result == record1
    assert f"Multiple records found for task {task_data['record_name']}" in caplog.text

def test_fetch_result_no_records(mock_direct_predict_record, task_data):
    mock_direct_predict_record.query_by_name.return_value = []
    
    task = DirectPredictTask(**task_data)
    result = task.fetch_result()
    
    mock_direct_predict_record.query_by_name.assert_called_once_with(task_data["record_name"])
    assert result is None

def test_sync_result_existing_record(mock_direct_predict_record, task_data, caplog):
    existing_record = MagicMock()
    mock_direct_predict_record.query_by_name.return_value = [existing_record]
    
    task = DirectPredictTask(**task_data)
    with caplog.at_level(logging.INFO):
        task.sync_result()
    
    assert f"TASK {task_data['record_name']} record found in database, SKIPPING." in caplog.text
    mock_direct_predict_record.query_by_name.assert_called_once_with(task_data["record_name"])


def test_sync_result_no_existing_record(mock_direct_predict_record, task_data, caplog):
    mock_direct_predict_record.query_by_name.return_value = []
    
    # Mock the run_task method
    with patch.object(DirectPredictTask, "run_task", return_value={"some_field": "some_value"}) as mock_run_task, caplog.at_level(logging.INFO):
        task = DirectPredictTask(**task_data)
        task.sync_result()
    
    mock_run_task.assert_called_once()
    assert f"TASK {task_data['record_name']} OUTPUT: {{'some_field': 'some_value'}}, INSERTING." in caplog.text
    mock_direct_predict_record.query_by_name.assert_called_once_with(task_data["record_name"])
    mock_direct_predict_record.assert_called_with(
        model_id="model1",
        record_name=task_data["record_name"],
        step="1000",
        task_name="taskA",
        some_field="some_value"
    )