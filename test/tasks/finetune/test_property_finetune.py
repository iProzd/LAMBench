from lambench.tasks.finetune.property_finetune import PropertyFinetuneTask
from unittest.mock import MagicMock, patch
import logging

def test_load_direct_predict_task(finetune_yml_data):
    for task_name, task_param in finetune_yml_data.items():
        model_id, step = "TEST_DP_v1", 1000
        record_name = f"{model_id}#{step}#{task_name}"
        task = PropertyFinetuneTask(record_name=record_name,**task_param)
        for key, value in task_param.items():
            assert getattr(task, key) == value


def test_fetch_result_single_record(mock_finetune_record, finetune_task_data):
    mock_record_instance = MagicMock()
    mock_finetune_record.query_by_name.return_value = [mock_record_instance]
    
    task = PropertyFinetuneTask(**finetune_task_data)
    result = task.fetch_result()
    mock_finetune_record.query_by_name.assert_called_once_with(finetune_task_data["record_name"])
    assert result == mock_record_instance


def test_fetch_result_multiple_records(mock_finetune_record, finetune_task_data, caplog):
    record1 = MagicMock()
    record2 = MagicMock()
    mock_finetune_record.query_by_name.return_value = [record1, record2]
    
    task = PropertyFinetuneTask(**finetune_task_data)
    with caplog.at_level(logging.WARNING):
        result = task.fetch_result()
    
    mock_finetune_record.query_by_name.assert_called_once_with(finetune_task_data["record_name"])
    assert result == record1
    assert f"Multiple records found for task {finetune_task_data['record_name']}" in caplog.text

def test_fetch_result_no_records(mock_finetune_record, finetune_task_data):
    mock_finetune_record.query_by_name.return_value = []
    
    task = PropertyFinetuneTask(**finetune_task_data)
    result = task.fetch_result()
    
    mock_finetune_record.query_by_name.assert_called_once_with(finetune_task_data["record_name"])
    assert result is None

def test_sync_result_existing_record(mock_finetune_record, finetune_task_data, caplog):
    existing_record = MagicMock()
    mock_finetune_record.query_by_name.return_value = [existing_record]
    
    task = PropertyFinetuneTask(**finetune_task_data)
    with caplog.at_level(logging.INFO):
        task.sync_result()
    
    assert f"TASK {finetune_task_data['record_name']} record found in database, SKIPPING." in caplog.text
    mock_finetune_record.query_by_name.assert_called_once_with(finetune_task_data["record_name"])


def test_sync_result_no_existing_record(mock_finetune_record, finetune_task_data, caplog):
    mock_finetune_record.query_by_name.return_value = []
    
    # Mock the run_task method
    with patch.object(PropertyFinetuneTask, "run_task", return_value={"some_field": "some_value"}) as mock_run_task, caplog.at_level(logging.INFO):
        task = PropertyFinetuneTask(**finetune_task_data)
        task.sync_result()
    
    mock_run_task.assert_called_once()
    assert f"TASK {finetune_task_data['record_name']} OUTPUT: {{'some_field': 'some_value'}}, INSERTING." in caplog.text
    mock_finetune_record.query_by_name.assert_called_once_with(finetune_task_data["record_name"])
    mock_finetune_record.assert_called_with(
        model_id="model1",
        record_name=finetune_task_data["record_name"],
        step="1000",
        task_name="taskA",
        some_field="some_value"
    )