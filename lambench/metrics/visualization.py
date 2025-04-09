import json
from pathlib import Path

import lambench
from lambench.metrics.vishelper.results_fetcher import ResultsFetcher
from lambench.metrics.vishelper.metrics_calculations import MetricsCalculator
from lambench.metrics.vishelper.plot_generation import PlotGeneration


class LAMBenchMetrics:
    def __init__(self):
        self.domain_results = ResultsFetcher()
        self.metrics_calculations = MetricsCalculator(self.domain_results)
        self.plot_generation = PlotGeneration(
            self.domain_results, self.metrics_calculations
        )

    def save_results(self):
        domain_results = self.domain_results.aggregate_ood_results()
        radar_chart_config = self.plot_generation.generate_radar_plot(domain_results)
        scatter_plot_data = self.plot_generation.generate_scatter_plot()
        barplot_data = self.plot_generation.generate_barplot(domain_results)
        final_ranking = self.metrics_calculations.summarize_final_rankings()

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


def main():
    metrics = LAMBenchMetrics()
    metrics.save_results()


if __name__ == "__main__":
    main()
