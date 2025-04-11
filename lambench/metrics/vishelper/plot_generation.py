import numpy as np


class PlotGeneration:
    def __init__(self, fetcher, metrics_calculator):
        self.fetcher = fetcher
        self.metrics_calculator = metrics_calculator

    def generate_radar_plot(self, domain_results: dict) -> dict:
        first_model = list(domain_results.keys())[0]
        categories = list(domain_results[first_model].keys())
        models = list(domain_results.keys())

        metrics_data = self._collect_metrics_data(domain_results, categories, models)
        normalized_metrics = self._normalize_metrics(
            domain_results, metrics_data["category_max"], categories
        )
        model_rankings = self._calculate_model_rankings(
            models, categories, metrics_data["category_values"]
        )
        best_model = self._find_best_model(model_rankings["total_rankings"])

        return self._build_radar_chart_config(
            categories, normalized_metrics, models, best_model
        )

    def generate_scatter_plot(self) -> list[dict]:
        results = []
        for model in self.fetcher.leaderboard_models:
            efficiency_raw = (
                self.fetcher.fetch_inference_efficiency_results_for_one_model(model)
            )
            zeroshot_raw = self.metrics_calculator.calculate_mean_m_bar_domain(model)
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

    def generate_barplot(self, domain_results: dict) -> dict:
        results = {}
        for model, domain_result in domain_results.items():
            for domain, metrics in domain_result.items():
                if domain not in results:
                    results[domain] = {}
                if metrics is not None:
                    results[domain][model] = np.round(metrics, 2)
        return results

    def _collect_metrics_data(
        self,
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
        self,
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
        self,
        models: list[str],
        categories: list[str],
        category_values: dict[str, list[float]],
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
            model: sum(
                category_rankings[category].get(model, 0) for category in categories
            )
            for model in models
        }
        return {
            "category_rankings": category_rankings,
            "total_rankings": total_rankings,
        }

    def _find_best_model(self, total_rankings: dict[str, int]) -> str | None:
        return min(total_rankings, key=total_rankings.get) if total_rankings else None

    def _build_radar_chart_config(
        self,
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
