import json
import logging
from pathlib import Path

import yaml

import lambench
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.databases.calculator_table import CalculatorRecord
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.workflow.entrypoint import gather_models
from lambench.metrics.utils import (
    filter_direct_task_results,
    exp_average,
    aggregated_nve_md_results,
)

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
    # Direct Task
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

    # Finetune Task
    if model.show_finetune_task:
        pass

    # Calculator Task
    if model.show_calculator_task:
        calculator_task_records = CalculatorRecord.query(model_name=model.model_name)
        if not calculator_task_records:
            logging.warning(f"No calculator task records found for {model.model_name}")
            return None

        calculator_task_results = {}
        # TODO aggregate results by tasks when more calculator tasks are added
        for record in calculator_task_records:
            calculator_task_results[record.task_name] = aggregated_nve_md_results(
                record.metrics
            )
        single_model_results["calculator_task_results"] = calculator_task_results

    return single_model_results


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
    json.dump(
        results, open(Path(__file__).parent / "results/results.json", "w"), indent=2
    )
    print("Results saved to results.json")


if __name__ == "__main__":
    main()
