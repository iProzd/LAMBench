import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import lambench
from lambench.databases.calculator_table import CalculatorRecord
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.metrics.post_process import DIRECT_TASK_WEIGHTS
from lambench.metrics.utils import (
    aggregated_inference_efficiency_results,
    aggregated_nve_md_results,
    exp_average,
    filter_direct_task_results,
    get_domain_to_direct_task_mapping,
    get_leaderboard_models,
)
from lambench.models.basemodel import BaseLargeAtomModel


def aggregate_domain_results_for_one_model(model: BaseLargeAtomModel):
    """
    This function fetch and process the raw results to calculate $\bar{M}_{\text{domain}}$ across all 5 domains for a single LAM.
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


def fetch_overall_zero_shot_results(model: BaseLargeAtomModel) -> float:
    """
    This function average $\bar{M}_{\text{domain}}$ over all 5 domains for a single LAM. result used in scatter plot.
    """
    domain_results = list(aggregate_domain_results_for_one_model(model).values())
    return np.mean(domain_results) if None not in domain_results else None


def fetch_generalizability_ood_results() -> dict[str, float]:
    leaderboard_models = get_leaderboard_models()
    generalizability_ood = {}
    for model in leaderboard_models:
        zero_shot_result = fetch_overall_zero_shot_results(model)
        if zero_shot_result is not None:
            generalizability_ood[model.model_metadata.pretty_name] = (
                1 - zero_shot_result
            )
    if not generalizability_ood:
        return None
    return generalizability_ood


def fetch_stability_results() -> dict[str, float]:
    leaderboard_models = get_leaderboard_models()
    stability_results = {}
    for model in leaderboard_models:
        task_results = CalculatorRecord.query(
            model_name=model.model_name, task_name="nve_md"
        )
        if len(task_results) != 1:
            logging.warning(
                f"Expected one record for {model.model_name} and nve_md, but got {len(task_results)}"
            )
            continue
        metrics = aggregated_nve_md_results(task_results[0].metrics)
        stability_results[model.model_metadata.pretty_name] = metrics

    if not stability_results:
        return None

    df = pd.DataFrame(stability_results).T
    epsilon = 0.5
    for col in ["std", "slope"]:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df[col] = epsilon + (1 - epsilon) * (df[col] - col_min) / (
                col_max - col_min
            )
    df["std"] /= df["success_rate"]
    df["slope"] /= df["success_rate"]
    df["stability_score"] = 1 - 0.5 * (df["std"] + df["slope"])
    score_min, score_max = df["stability_score"].min(), df["stability_score"].max()
    if score_max > score_min:
        df["stability_score"] = (df["stability_score"] - score_min) / (
            score_max - score_min
        )
    return df["stability_score"].to_dict()


def fetch_applicability_results() -> dict[str, float]:
    leaderboard_models = get_leaderboard_models()
    efficiency_results = {}
    for model in leaderboard_models:
        efficiency_raw = fetch_inference_efficiency_results(model)
        if efficiency_raw is None or efficiency_raw["average_time"] is None:
            continue
        efficiency_results[model.model_metadata.pretty_name] = np.round(
            efficiency_raw["average_time"], 2
        )
    if not efficiency_results:
        return None

    df_eff = pd.DataFrame.from_dict(
        efficiency_results, orient="index", columns=["inference_time"]
    )
    df_eff["efficiency_score"] = 1 - (
        df_eff["inference_time"] - df_eff["inference_time"].min()
    ) / (df_eff["inference_time"].max() - df_eff["inference_time"].min())

    stability_results = fetch_stability_results()
    shared_models = set(df_eff.index).intersection(set(stability_results.keys()))
    if not shared_models:
        return None

    applicability_results = {
        model: (df_eff.loc[model]["efficiency_score"] + stability_results[model]) / 2
        for model in shared_models
    }
    return applicability_results


def fetch_inference_efficiency_results(model: BaseLargeAtomModel) -> dict[str, float]:
    task_results = CalculatorRecord.query(
        model_name=model.model_name, task_name="inference_efficiency"
    )
    if len(task_results) != 1:
        logging.warning(
            f"Expected one record for {model.model_name} and inference_efficiency, but got {len(task_results)}"
        )
        return None
    return aggregated_inference_efficiency_results(task_results[0].metrics)


def aggregate_domain_results() -> dict[str, dict[str, float]]:
    results = {}
    leaderboard_models = get_leaderboard_models()
    for model in leaderboard_models:
        results[model.model_metadata.pretty_name] = (
            aggregate_domain_results_for_one_model(model)
        )
    return results


def generate_radar_plot(domain_results: dict) -> dict:
    first_model = list(domain_results.keys())[0]
    categories = list(domain_results[first_model].keys())
    models = list(domain_results.keys())

    metrics_data = _collect_metrics_data(domain_results, categories, models)
    normalized_metrics = _normalize_metrics(
        domain_results, metrics_data["category_max"], categories
    )
    model_rankings = _calculate_model_rankings(
        models, categories, metrics_data["category_values"]
    )
    best_model = _find_best_model(model_rankings["total_rankings"])

    return _build_radar_chart_config(categories, normalized_metrics, models, best_model)


def generate_scatter_plot() -> list[dict]:
    results = []
    leaderboard_models = get_leaderboard_models()
    for model in leaderboard_models:
        efficiency_raw = fetch_inference_efficiency_results(model)
        zeroshot_raw = fetch_overall_zero_shot_results(model)
        if (
            efficiency_raw is None
            or efficiency_raw["average_time"] is None
            or zeroshot_raw is None
        ):
            continue
        results.append(
            {
                "name": model.model_metadata.pretty_name,
                "family": model.model_family,
                "nparams": model.model_metadata.num_parameters,
                "efficiency": np.round(efficiency_raw["average_time"], 2),
                "std": np.round(efficiency_raw["standard_deviation"], 2),
                "zeroshot": np.round(zeroshot_raw, 2),
            }
        )
    return results


def generate_barplot(domain_results: dict) -> dict:
    results = {}
    for model, domain_result in domain_results.items():
        for domain, metrics in domain_result.items():
            if domain not in results:
                results[domain] = {}
            if metrics is not None:
                results[domain][model] = np.round(metrics, 2)
    return results


def _collect_metrics_data(
    domain_results: dict[str, dict[str, float]],
    categories: list[str],
    models: list[str],
) -> dict[str, dict]:
    category_values = {category: [] for category in categories}
    for model in models:
        for category in categories:
            if domain_results[model][category]:
                category_values[category].append(
                    -np.log(domain_results[model][category])
                )
    category_max = {
        category: max(values) if values else 1.0
        for category, values in category_values.items()
    }
    return {"category_values": category_values, "category_max": category_max}


def _normalize_metrics(
    domain_results: dict[str, dict[str, float]],
    category_max: dict[str, float],
    categories: list[str],
) -> dict[str, list[float | None]]:
    normalized_metrics = {}
    for model, res in domain_results.items():
        normalized_metrics[model] = []
        for category in categories:
            if res[category]:
                normalized_value = (-np.log(res[category])) / category_max[category]
                normalized_metrics[model].append(normalized_value)
            else:
                normalized_metrics[model].append(None)
    return normalized_metrics


def _calculate_model_rankings(
    models: list[str], categories: list[str], category_values: dict[str, list[float]]
) -> dict[str, dict]:
    category_rankings = {}
    for category in categories:
        category_rankings[category] = {}
        values = category_values[category]
        if values:
            sorted_values = sorted(values, reverse=True)
            model_values = {
                model: value
                for model, value in zip(models, values)
                if value is not None
            }
            for model, value in model_values.items():
                rank = sorted_values.index(value) + 1
                category_rankings[category][model] = rank
    total_rankings = {
        model: sum(category_rankings[category].get(model, 0) for category in categories)
        for model in models
    }
    return {"category_rankings": category_rankings, "total_rankings": total_rankings}


def _find_best_model(total_rankings: dict[str, int]) -> str | None:
    return min(total_rankings, key=total_rankings.get) if total_rankings else None


def _build_radar_chart_config(
    categories: list[str],
    normalized_metrics: dict[str, list[float | None]],
    models: list[str],
    best_model: str | None,
    text_color: str = "white",
) -> dict:
    area_style = {"areaStyle": {"opacity": 0.1}}
    chart_config = {
        "legend": {"data": models, "bottom": 0, "textStyle": {"color": text_color}},
        "radar": {
            "indicator": [{"name": category, "max": 1} for category in categories],
            "axisName": {"color": text_color},
        },
        "series": [
            {
                "name": "LAMBench Leaderboard",
                "type": "radar",
                "data": [
                    {"name": model, "value": values}
                    for model, values in normalized_metrics.items()
                ],
            }
        ],
    }
    if best_model:
        for model_data in chart_config["series"][0]["data"]:
            if model_data["name"] == best_model:
                model_data.update(area_style)
    return chart_config


def summarize_final_rankings():
    generalizability_ood = fetch_generalizability_ood_results()
    applicability = fetch_applicability_results()
    # Skip if either result set is empty
    if not generalizability_ood or not applicability:
        logging.warning("Missing data for generalizability or applicability metrics")
        return

    # Find models that appear in both metrics
    shared_models = set(generalizability_ood.keys()).intersection(
        set(applicability.keys())
    )

    if not shared_models:
        logging.warning(
            "No models have both generalizability and applicability metrics"
        )
        return

    # Create a dataframe with both metrics
    summary_df = pd.DataFrame(
        {
            "model": list(shared_models),
            "generalizability": [
                generalizability_ood[model] for model in shared_models
            ],
            "applicability": [applicability[model] for model in shared_models],
        }
    )

    summary_df = summary_df.sort_values("generalizability")

    # Calculate overall ranking (you can adjust the weights as needed)
    summary_df["overall_score"] = (
        0.5 * summary_df["generalizability"] + 0.5 * summary_df["applicability"]
    )

    # Sort by overall score for final ranking
    summary_df = summary_df.sort_values("overall_score", ascending=False)
    summary_df.reset_index(drop=True, inplace=True)
    summary_df["rank"] = summary_df.index + 1
    summary_df = summary_df[["rank", "model", "generalizability", "applicability"]]
    summary_df.columns = ["Rank", "Model", "Generalizability", "Applicability"]
    summary_df = summary_df.round(3)

    return summary_df


def main():
    domain_results = aggregate_domain_results()
    radar_chart_config = generate_radar_plot(domain_results)
    scatter_plot_data = generate_scatter_plot()
    barplot_data = generate_barplot(domain_results)
    final_ranking = summarize_final_rankings()

    result_path = Path(lambench.__file__).parent / "metrics/results"
    with open(result_path / "radar.json", "w") as f:
        json.dump(radar_chart_config, f, indent=2)
    with open(result_path / "scatter.json", "w") as f:
        json.dump(scatter_plot_data, f, indent=2)
    with open(result_path / "barplot.json", "w") as f:
        json.dump(barplot_data, f, indent=2)
    with open(result_path / "final_rankings.json", "w") as f:
        json.dump(final_ranking.to_dict(orient="records"), f, indent=2)
    print("All plots saved to metrics/results/")


if __name__ == "__main__":
    main()
