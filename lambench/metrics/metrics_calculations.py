import logging
import numpy as np
import pandas as pd
from lambench.databases.calculator_table import CalculatorRecord
from lambench.metrics.utils import (
    aggregated_inference_efficiency_results,
    aggregated_nve_md_results,
)


class MetricsCalculations:
    def __init__(self, domain_results):
        self.domain_results = domain_results

    def fetch_overall_zero_shot_results(self, model) -> float:
        domain_results = list(
            self.domain_results.aggregate_domain_results_for_one_model(model).values()
        )
        return np.mean(domain_results) if None not in domain_results else None

    def fetch_generalizability_ood_results(self) -> dict[str, float]:
        generalizability_ood = {}
        for model in self.domain_results.leaderboard_models:
            zero_shot_result = self.fetch_overall_zero_shot_results(model)
            if zero_shot_result is not None:
                generalizability_ood[model.model_metadata.pretty_name] = (
                    1 - zero_shot_result
                )
        return generalizability_ood if generalizability_ood else None

    def fetch_stability_results(self) -> dict[str, float]:
        stability_results = {}
        for model in self.domain_results.leaderboard_models:
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

    def fetch_applicability_results(self) -> dict[str, float]:
        efficiency_results = {}
        for model in self.domain_results.leaderboard_models:
            efficiency_raw = self.fetch_inference_efficiency_results(model)
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

        stability_results = self.fetch_stability_results()
        shared_models = set(df_eff.index).intersection(set(stability_results.keys()))
        if not shared_models:
            return None

        applicability_results = {
            model: (df_eff.loc[model]["efficiency_score"] + stability_results[model])
            / 2
            for model in shared_models
        }
        return applicability_results

    def fetch_inference_efficiency_results(self, model) -> dict[str, float]:
        task_results = CalculatorRecord.query(
            model_name=model.model_name, task_name="inference_efficiency"
        )
        if len(task_results) != 1:
            logging.warning(
                f"Expected one record for {model.model_name} and inference_efficiency, but got {len(task_results)}"
            )
            return None
        return aggregated_inference_efficiency_results(task_results[0].metrics)

    def summarize_final_rankings(self):
        generalizability_ood = self.fetch_generalizability_ood_results()
        applicability = self.fetch_applicability_results()
        if not generalizability_ood or not applicability:
            logging.warning(
                "Missing data for generalizability or applicability metrics"
            )
            return

        shared_models = set(generalizability_ood.keys()).intersection(
            set(applicability.keys())
        )

        if not shared_models:
            logging.warning(
                "No models have both generalizability and applicability metrics"
            )
            return

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

        summary_df["overall_score"] = (
            0.5 * summary_df["generalizability"] + 0.5 * summary_df["applicability"]
        )

        summary_df = summary_df.sort_values("overall_score", ascending=False)
        summary_df.reset_index(drop=True, inplace=True)
        summary_df["rank"] = summary_df.index + 1
        summary_df = summary_df[["rank", "model", "generalizability", "applicability"]]
        summary_df.columns = ["Rank", "Model", "Generalizability", "Applicability"]
        summary_df = summary_df.round(3)

        return summary_df
