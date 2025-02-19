import logging
from lambench.metrics.post_process import DIRECT_TASK_WEIGHTS
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.metrics.utils import get_domain_to_direct_task_mapping
from lambench.metrics.utils import filter_direct_task_results, exp_average
from lambench.workflow.entrypoint import gather_models


def aggregate_domain_results_for_one_model(model: BaseLargeAtomModel):
    """
    This function aggregates the results for one model across domains.
    """

    domain_results = {}
    for domain, tasks in get_domain_to_direct_task_mapping(DIRECT_TASK_WEIGHTS).items():
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
            domain_results[domain] = exp_average(norm_log_results)

            # aggregate over E, F, V, TODO refactor
            nomarlized_e = domain_results[domain]["energy_mae_natoms"]
            nomarlized_f = domain_results[domain]["force_mae"]
            nomarlized_v = domain_results[domain]["virial_mae_natoms"]
            if weight_virial:
                domain_results[domain] = (
                    0.45 * nomarlized_e + 0.45 * nomarlized_f + 0.1 * nomarlized_v
                )
            else:
                domain_results[domain] = 0.5 * nomarlized_e + 0.5 * nomarlized_f
    return domain_results


def aggregate_domain_results():
    """
    This function aggregates the results across models and domains.
    """
    results = {}
    models = gather_models()
    leaderboard_models = [
        model
        for model in models
        if model.show_direct_task
        or model.show_finetune_task
        or model.show_calculator_task
    ]

    for model in leaderboard_models:
        results[model.model_name] = aggregate_domain_results_for_one_model(model)

    return results


if __name__ == "__main__":
    print(aggregate_domain_results())
