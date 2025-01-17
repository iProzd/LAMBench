import logging
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

import lambench
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.workflow.entrypoint import gather_models

DIRECT_TASK_METRICS = {
    k: v
    for k, v in yaml.safe_load(
        open(Path(lambench.__file__).parent / "metrics/direct_tasks_metrics.yml", "r")
    ).items()
}


def process_results_for_one_model(model: BaseLargeAtomModel):
    """
    This function fetch and process the raw results from corresponding tables for one model across required tasks.
    """
    single_model_results = {
        "model_name": model.model_name,
        "direct_task_results": {},
        "finetune_task_results": {},
        "calculator_task_results": {},
    }

    if model.show_direct_task:
        direct_task_records = DirectPredictRecord.query(model_name=model.model_name)
        if not direct_task_records:
            logging.warning(f"No direct task records found for {model.model_name}")
            return None

        direct_task_results = {}
        normalized_results = []
        for record in direct_task_records:
            direct_task_results[record.task_name] = record.__dict__
            normalized_result = filter_direct_task_results(
                record.__dict__, DIRECT_TASK_METRICS[record.task_name]
            )
            # NOTE: __dict__ contains sqlalchemy objects and non-metric info.
            normalized_results.append(normalized_result)

        direct_task_results["Weighted"] = exp_average(normalized_results)
        single_model_results["direct_task_results"] = direct_task_results

    if model.show_finetune_task:
        pass
    if model.show_calculator_task:
        raise NotImplementedError("Calculator task is not implemented yet.")

    return single_model_results


def filter_direct_task_results(
    task_result: dict, task_config: dict
) -> dict:
    """
    This function filters the direct task results to keep only the metrics with non-zero task weights.

    I. Optional: normalize the metrics by multiply {metric}_std. (Required for Property)
    II. Remove tasks where weight is None in the DIRECT_TASK_METRICS.

    NOTE: We normalize first to ensure the weight is a dimensionless number.

    Returns: metrics for each task normalized and weighted.
    """
    filtered_metrics = {}
    metrics = ["energy", "force", "virial"]
    for metric in metrics:
        # Set default value to None
        filtered_metrics[f"{metric}_mae"] = None
        filtered_metrics[f"{metric}_rmse"] = None
        if metric != "force":
            filtered_metrics[f"{metric}_mae_natoms"] = None
            filtered_metrics[f"{metric}_rmse_natoms"] = None

        weight = task_config.get(f"{metric}_weight")
        if weight is None:
            continue
        std = task_config.get(f"dataset_lstsq_std_{metric}")
        normalize = True # TODO: make it configurable
        if normalize and std is not None:
            weight /= std
        filtered_metrics[f"{metric}_mae"] = task_result[f"{metric}_mae"] * weight
        filtered_metrics[f"{metric}_rmse"] = task_result[f"{metric}_rmse"] * weight
        if metric != "force":
            filtered_metrics[f"{metric}_mae_natoms"] = task_result[f"{metric}_mae_natoms"]
            filtered_metrics[f"{metric}_rmse_natoms"] =  task_result[f"{metric}_rmse_natoms"]

    return filtered_metrics


def exp_average(results: list[dict]) -> dict[str, Optional[float]]:
    """Calculate the exponential average of each metric of the results.
    """
    exp_average_metrics = {}
    for key, value in results[0].items():  # use key and value from the first result
        if (
            isinstance(value, float)
            or value is None  # unluckily got None for results[0]
        ):
            try:
                metrics_list = [result[key] for result in results]
                exp_average_metrics[key] = np.exp(np.mean(np.log(metrics_list)))
            except TypeError:
                # Contains None(NaN); for the comparability among tasks, set it to None
                exp_average_metrics[key] = None
    return exp_average_metrics


def main():
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
        results[model.model_name] = process_results_for_one_model(model)

        # DEBUG: print the weighted results
        r = results[model.model_name]
        assert r is not None
        print(model.model_name, r["direct_task_results"]["Weighted"])

if __name__ == "__main__":
    main()