import logging
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.metrics.post_process import DIRECT_TASK_WEIGHTS
from lambench.metrics.utils import (
    exp_average,
    filter_direct_task_results,
    get_domain_to_direct_task_mapping,
    get_leaderboard_models,
)
from lambench.models.basemodel import BaseLargeAtomModel


class DomainResults:
    def __init__(self):
        self.leaderboard_models = get_leaderboard_models()

    def aggregate_domain_results_for_one_model(self, model: BaseLargeAtomModel) -> dict:
        domain_results = {}
        for domain, tasks in get_domain_to_direct_task_mapping(
            DIRECT_TASK_WEIGHTS
        ).items():
            norm_log_results = []
            weight_virial = False
            for task in tasks:
                task_result = DirectPredictRecord.query(
                    model_name=model.model_name, task_name=task
                )
                task_config = DIRECT_TASK_WEIGHTS[task]
                if task_config["virial_weight"] is not None:
                    weight_virial = True
                if len(task_result) != 1:
                    logging.warning(
                        f"Expect one record for {model.model_name} and {task}, but got {len(task_result)}"
                    )
                    continue
                norm_log_results.append(
                    filter_direct_task_results(
                        task_result[0].to_dict(), task_config, normalize=True
                    )
                )
            if len(norm_log_results) != len(tasks):
                domain_results[domain] = None
            else:
                domain_result = exp_average(norm_log_results)
                normalized_e = domain_result["energy_mae_natoms"]
                normalized_f = domain_result["force_mae"]
                normalized_v = domain_result["virial_mae_natoms"]
                domain_results[domain] = (
                    0.45 * normalized_e + 0.45 * normalized_f + 0.1 * normalized_v
                    if weight_virial
                    else 0.5 * normalized_e + 0.5 * normalized_f
                )
        return domain_results

    def aggregate_domain_results(self) -> dict[str, dict[str, float]]:
        results = {}
        for model in self.leaderboard_models:
            results[model.model_metadata.pretty_name] = (
                self.aggregate_domain_results_for_one_model(model)
            )
        return results
