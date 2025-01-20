import json
import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import yaml

import lambench
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.workflow.entrypoint import gather_models

DIRECT_TASK_METRICS = {
    k: v
    for k, v in yaml.safe_load(
        open(Path(lambench.__file__).parent / "metrics/direct_task_weights.yml", "r")
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
        norm_log_results = []
        for record in direct_task_records:
            direct_task_results[record.task_name] = record.to_dict()
            normalized_result = filter_direct_task_results(
                record.to_dict(), DIRECT_TASK_METRICS[record.task_name]
            )
            norm_log_results.append(normalized_result)

        direct_task_results["Weighted"] = exp_average(norm_log_results)
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
    III. Calculate the weighted **log** metrics.

    NOTE: We normalize first to ensure the weight is a dimensionless number.

    Returns: metrics for each task normalized, logged, and weighted.
    """
    filtered_metrics = {}
    for k, v in task_result.items():
        efv: Literal["energy", "force", "virial"] = k.split("_")[0]
        weight = task_config.get(f"{efv}_weight")
        if weight is None:
            filtered_metrics[k] = None
            continue
        std = task_config.get(f"{efv}_std")
        normalize = True  # TODO: make it configurable
        if normalize and std is not None:
            weight /= std
        filtered_metrics[k] = np.log(v) * weight
    return filtered_metrics


def exp_average(log_results: list[dict]) -> dict[str, Optional[float]]:
    """Calculate the exponential average of each metric of the results.
    """
    exp_average_metrics = {}
    for key, value in log_results[0].items():  # use key and value from the first result
        if (
            isinstance(value, float)
            or value is None  # unluckily got None for results[0]
        ):
            try:
                metrics_list = [result[key] for result in log_results]
                exp_average_metrics[key] = np.exp(np.mean(metrics_list))
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
    json.dump(results, open("results.json", "w"), indent=2)
    print("Results saved to results.json")
if __name__ == "__main__":
    main()