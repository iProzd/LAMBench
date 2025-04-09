import logging
from typing import Optional
import numpy as np
import pandas as pd
from collections import defaultdict
from lambench.metrics.vishelper.results_fetcher import DOWNSTREAM_TASK_METRICS


class MetricsCalculator:
    def __init__(self, fetcher):
        self.fetcher = fetcher

    def calculate_mean_m_bar_domain(self, model) -> float:
        """This calculates $\bar{M}_{\text{domain}}$ for a given LAM across all domains."""
        domain_results = list(
            self.fetcher.aggregate_ood_results_for_one_model(model).values()
        )
        return np.mean(domain_results) if None not in domain_results else None

    def convert_metric_to_score(
        self,
        metric_dict: dict[str, float],
        method: str = "minmax",
        dummy: Optional[float] = None,
    ) -> dict[str, float]:
        """Convert metric values (where lower is better) to normalized scores in range [0, 1] (where higher is better)."""
        if not metric_dict:
            return {}
        scores = {}
        if method == "minmax":
            min_value = min(metric_dict.values())
            if dummy is not None:
                max_value = dummy
            else:
                max_value = max(metric_dict.values())

            # If all values are the same, return 1.0 for all
            if max_value == min_value:
                return {model: 1.0 for model in metric_dict}
            # Convert: score = 1 - (value - min) / (max - min)
            for model, value in metric_dict.items():
                scores[model] = 1.0 - (value - min_value) / (max_value - min_value)
        elif method == "-log":
            max_value = max([-np.log(value) for value in metric_dict.values()])
            for model, value in metric_dict.items():
                if value == 0:
                    scores[model] = 1
                else:
                    scores[model] = -np.log(value) / max_value
        else:
            raise ValueError(f"Unknown method: {method}")
        return scores

    def calculate_generalizability_ood_score(self) -> dict[str, float]:
        """Calculate generalizability score for each model across all domains.

        Returns a dictionary mapping model names to scores in [0,1] range.
        0 indicates baseline performance, 1 indicates best performance across all domains.
        """
        # Get model performances by domain
        m_bar_domain = self.fetcher.aggregate_ood_results()
        # filter out models with missing domain results
        m_bar_domain = {
            model: domains
            for model, domains in m_bar_domain.items()
            if None not in domains.values()
        }

        if not m_bar_domain:
            logging.warning("No domain results found.")
            return {}

        # Reorganize data {model: {domain: value}} to {domain: {model: value}}
        reorg_m_bar_domain = defaultdict(dict)
        for model, domains in m_bar_domain.items():
            for domain, value in domains.items():
                reorg_m_bar_domain[domain][model] = value

        s_domain = {
            domain: self.convert_metric_to_score(metrics, method="-log")
            for domain, metrics in reorg_m_bar_domain.items()
        }
        print("s_domain", s_domain)

        # Calculate average score across all domains for each model
        mean_s_domain = defaultdict(list)
        for domain_scores in s_domain.values():
            for model, score in domain_scores.items():
                mean_s_domain[model].append(score)

        # Calculate final generalizability score as mean across domains
        generalizability_scores = {
            model: np.mean(scores)
            for model, scores in mean_s_domain.items()
            if scores  # Skip models with no scores
        }

        return generalizability_scores

    def calculate_generalizability_downstream_score(self) -> dict[str, float]:
        raw_results = self.fetcher.fetch_downstream_results()
        penalty_dict = {}
        for task_name, task_config in DOWNSTREAM_TASK_METRICS.items():
            if task_config.get("penalty", None) is not None:
                penalty_dict[f'{task_name}::{task_config["penalty"]}'] = [
                    f"{task_name}::{metrics_name}"
                    for metrics_name in task_config["metrics"]
                ]

        # apply minmax normalization for all metrics, use dummy when available
        for column in raw_results.columns:
            if column not in penalty_dict:
                check_dummy = column.split(
                    "::"
                )  # split the column name back to task name and metric name
                assert (
                    len(check_dummy) == 2
                ), f"Column name {column} is not in the expected format"
                # extract dummy max value when available
                if (
                    DOWNSTREAM_TASK_METRICS[check_dummy[0]].get("dummy", None)
                    is not None
                ):
                    dummy = DOWNSTREAM_TASK_METRICS[check_dummy[0]]["dummy"][
                        check_dummy[1]
                    ]
                else:
                    dummy = None
                raw_results[column] = self.convert_metric_to_score(
                    raw_results[column].to_dict(), method="minmax", dummy=dummy
                )

        # apply penalty for each task
        for task_name, penalty in penalty_dict.items():
            for penalty_metric in penalty:
                if penalty_metric not in raw_results.columns:
                    logging.error(f"Metric {penalty_metric} not found in raw results")
                    continue
                # apply penalty
                # Apply penalty based on sign of the metric
                positive_mask = raw_results[penalty_metric] >= 0
                negative_mask = raw_results[penalty_metric] < 0

                # Create a copy to avoid SettingWithCopyWarning
                new_values = raw_results[penalty_metric].copy()

                # For positive values, multiply
                new_values[positive_mask] = (
                    raw_results[penalty_metric][positive_mask]
                    * raw_results[task_name][positive_mask]
                )

                # For negative values, divide
                new_values[negative_mask] = (
                    raw_results[penalty_metric][negative_mask]
                    / raw_results[task_name][negative_mask]
                )

                raw_results[penalty_metric] = new_values

        # aggregate all metrics for each model
        downstream_score = raw_results.drop(columns=penalty_dict.keys()).mean(axis=1)
        return downstream_score.to_dict()

    def calculate_stability_results(self) -> dict[str, float]:
        """This calculates the stability score for a given LAM."""
        stability_results = self.fetcher.fetch_stability_results()
        # filter out models with missing stability results
        stability_results = {
            model: metrics
            for model, metrics in stability_results.items()
            if metrics is not None and metrics["success_rate"] > 0
        }
        if not stability_results:
            logging.warning("No stability results found.")
            return {}

        # reorganize data {model: {metric: value}} to {metric: {model: value}}
        stability_results_reorg = defaultdict(dict)
        for model, metrics in stability_results.items():
            for metric, value in metrics.items():
                stability_results_reorg[metric][model] = value

        # apply minmax normalization
        raw_stability_scores = {}
        for metric, models in stability_results_reorg.items():
            if metric == "success_rate":
                continue
            raw_stability_scores[metric] = self.convert_metric_to_score(
                models, method="minmax"
            )

        # compute final stability score
        stability_scores = {}
        for model in stability_results:
            stability_scores[model] = (
                raw_stability_scores["std"][model] * 0.5
                + raw_stability_scores["slope"][model] * 0.5
            ) * stability_results[model]["success_rate"]  # penalty for success rate
        return stability_scores

    def calculate_efficiency_results(self) -> dict[str, float]:
        efficiency_results = self.fetcher.fetch_inference_efficiency_results()
        # filter out models with missing efficiency results
        efficiency_results = {
            model: metrics
            for model, metrics in efficiency_results.items()
            if metrics is not None and metrics["average_time"] is not None
        }
        if not efficiency_results:
            logging.warning("No inference efficiency results found.")
            return {}
        # extract inference time and calculate efficiency score

        efficiency_scores = {}
        for model, metrics in efficiency_results.items():
            if metrics["average_time"] is None:
                continue
            efficiency_scores[model] = metrics["average_time"]
        efficiency_scores = self.convert_metric_to_score(
            efficiency_scores, method="minmax"
        )
        return efficiency_scores

    def calculate_applicability_results(self) -> dict[str, float]:
        """This function summarizes the applicability results for all models."""

        efficiency_results = self.calculate_efficiency_results()
        stability_results = self.calculate_stability_results()
        shared_models = set(efficiency_results.keys()).intersection(
            set(stability_results.keys())
        )
        if not shared_models:
            logging.warning("No models have both efficiency and stability metrics")
            return {}
        applicability_scores = {}
        for model in shared_models:
            applicability_scores[model] = (
                0.5 * efficiency_results[model] + 0.5 * stability_results[model]
            )
        return applicability_scores

    def summarize_final_rankings(self):
        generalizability_ood = self.calculate_generalizability_ood_score()
        generalizability_downstream = self.calculate_generalizability_downstream_score()
        if not generalizability_ood or not generalizability_downstream:
            logging.warning(
                "Missing data for generalizability metrics (ood or downstream)"
            )
            return
        applicability = self.calculate_applicability_results()
        if not generalizability_ood or not applicability:
            logging.warning(
                "Missing data for generalizability or applicability metrics"
            )
            return

        shared_models = (
            set(generalizability_ood.keys())
            .intersection(set(applicability.keys()))
            .intersection(set(generalizability_downstream.keys()))
        )
        if not shared_models:
            logging.warning(
                "No models have both generalizability and applicability metrics"
            )
            return

        summary_df = pd.DataFrame(
            {
                "model": list(shared_models),
                "generalizability-ood": [
                    generalizability_ood[model] for model in shared_models
                ],
                "generalizability-downstream": [
                    generalizability_downstream[model] for model in shared_models
                ],
                "applicability": [applicability[model] for model in shared_models],
            }
        )

        summary_df["overall_score"] = summary_df[
            ["generalizability-ood", "generalizability-downstream", "applicability"]
        ].mean(axis=1)

        summary_df = summary_df.sort_values("overall_score", ascending=False)
        summary_df.reset_index(drop=True, inplace=True)
        summary_df["rank"] = summary_df.index + 1
        summary_df = summary_df[
            [
                "rank",
                "model",
                "generalizability-ood",
                "generalizability-downstream" "applicability",
            ]
        ]
        summary_df.columns = [
            "Rank",
            "Model",
            "Generalizability-OOD",
            "Generalizability-Downstream",
            "Applicability",
        ]
        summary_df = summary_df.round(3)
        return summary_df
