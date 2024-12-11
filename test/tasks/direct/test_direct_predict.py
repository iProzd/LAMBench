from lambench.tasks.direct.direct_predict import DirectPredictTask
import yaml

def test_load_direct_predict_task(direct_predict_task_data):
    
    for task_name, task_param in direct_predict_task_data.items():
        task = DirectPredictTask(task_name=task_name,**task_param)
        assert task.test_data is not None
        assert task.result is None
        assert task.energy_weight == task_param["energy_weight"]
        assert task.force_weight == task_param["force_weight"]
        assert task.virial_weight == task_param.get("virial_weight")
