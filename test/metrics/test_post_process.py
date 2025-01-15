from lambench.metrics.post_process import LeaderboardModel
import yaml


def test_LeaderboardModels():
    with open("lambench/metrics/models_to_show.yml", "r") as file:
        yaml_data = yaml.safe_load(file)
    errors = []
    for model, model_config in yaml_data.items():
        try:
            LeaderboardModel(**model_config)
        except ValueError as e:
            errors.append(e)
    assert len(errors) == 0
