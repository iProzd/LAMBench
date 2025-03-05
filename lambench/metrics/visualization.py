import logging
from typing import Optional
from lambench.metrics.post_process import DIRECT_TASK_WEIGHTS
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.databases.calculator_table import CalculatorRecord
from lambench.metrics.utils import (
    get_domain_to_direct_task_mapping,
    aggregated_nve_md_results,
    aggregated_inference_efficiency_results,
    filter_direct_task_results,
    exp_average,
)
from lambench.workflow.entrypoint import gather_models
import numpy as np


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
            normalized_e = domain_results[domain]["energy_mae_natoms"]
            normalized_f = domain_results[domain]["force_mae"]
            normalized_v = domain_results[domain]["virial_mae_natoms"]
            if weight_virial:
                domain_results[domain] = (
                    0.45 * normalized_e + 0.45 * normalized_f + 0.1 * normalized_v
                )
            else:
                domain_results[domain] = 0.5 * normalized_e + 0.5 * normalized_f
    return domain_results


def fetch_stability_results(model: BaseLargeAtomModel) -> Optional[float]:
    """
    Fetch stability results from NVE MD task for a given model.

    The stability metric is calculated as (slope + std) / 2 - log(success_rate) / 100.
    - 'slope': energy drift slope in molecular dynamics simulation
    - 'std': standard deviation of energy during simulation
    - 'success_rate': percentage of successful simulation runs

    Lower values indicate better stability, with penalties for failed simulations.

    Returns:
        float: Combined stability metric, or None if results are not available
    """
    task_results = CalculatorRecord.query(
        model_name=model.model_name, task_name="nve_md"
    )

    if len(task_results) != 1:
        logging.warning(
            f"Expected one record for {model.model_name} and nve_md, but got {len(task_results)}"
        )
        return None

    metrics = aggregated_nve_md_results(task_results[0].metrics)
    slope = metrics["slope"]
    std = metrics["std"]
    success_rate = metrics["success_rate"]

    return (slope + std) / 2 - np.log(
        success_rate
    ) / 100  # to penalize failed simulations


def fetch_inference_efficiency_results(model: BaseLargeAtomModel) -> Optional[float]:
    task_results = CalculatorRecord.query(
        model_name=model.model_name, task_name="inference_efficiency"
    )

    if len(task_results) != 1:
        logging.warning(
            f"Expected one record for {model.model_name} and inference_efficiency, but got {len(task_results)}"
        )
        return None

    metrics = aggregated_inference_efficiency_results(task_results[0].metrics)
    return metrics["average_time_per_step"]


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
        domain_results = aggregate_domain_results_for_one_model(model)
        stability = fetch_stability_results(model)
        domain_results["Stability"] = stability
        inference_efficiency = fetch_inference_efficiency_results(model)
        domain_results["Efficiency"] = inference_efficiency
        results[model.model_name] = domain_results

    return results
