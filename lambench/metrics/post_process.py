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

DIRECT_TASK_WEIGHTS = {
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
                direct_task_results[record.task_name],
                DIRECT_TASK_WEIGHTS[record.task_name],
            )
            norm_log_results.append(normalized_result)

        if len(direct_task_records) != len(DIRECT_TASK_WEIGHTS):
            direct_task_results["Weighted"] = None
            missing_tasks = DIRECT_TASK_WEIGHTS.keys() - direct_task_results.keys()
            logging.warning(
                f"Weighted results for {model.model_name} are marked as None due to missing tasks: {missing_tasks}"
            )
        else:
            direct_task_results["Weighted"] = exp_average(norm_log_results)
        single_model_results["direct_task_results"] = direct_task_results

    if model.show_finetune_task:
        pass
    if model.show_calculator_task:
        raise NotImplementedError("Calculator task is not implemented yet.")

    return single_model_results


def filter_direct_task_results(task_result: dict, task_config: dict) -> dict:
    """
    This function filters the direct task results to keep only the metrics with non-zero task weights.

    I. Optional: normalize the metrics by multiply {metric}_std. (Required for Property)
    II. Remove tasks where weight is None in the DIRECT_TASK_METRICS.
        Please note that this change also applies in the input dict.
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
            task_result[k] = None
            continue
        std = task_config.get(f"{efv}_std")
        normalize = True  # TODO: make it configurable
        if normalize and std is not None:
            weight /= std

        if v is not None:
            filtered_metrics[k] = np.log(v) * weight
            # else the filtered_metrics will not have this key.
            # Metrics with weight != None should have a value,
            # Or the weighted result would be marked to None.
    return filtered_metrics


def exp_average(log_results: list[dict]) -> dict[str, Optional[float]]:
    """Calculate the exponential average of each metric of the results."""
    exp_average_metrics = {}
    all_keys = set([key for result in log_results for key in result.keys()])
    for key in sorted(all_keys):
        try:
            metrics_list = [result[key] for result in log_results]
        except KeyError:
            # Contains None(NaN) for metrics with weight != None;
            # For the comparability among tasks, set it to None
            exp_average_metrics[key] = None
            continue
        # Filter out "legal" None values with weight == None
        metrics_list = [m for m in metrics_list if m is not None]
        exp_average_metrics[key] = np.exp(np.mean(metrics_list))
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
        # PosixPath is not JSON serializable
        results[model.model_name]["model"] = model.model_dump(exclude={"model_path"})
    json.dump(results, open("results.json", "w"), indent=2)
    print("Results saved to results.json")


if __name__ == "__main__":
    main()
