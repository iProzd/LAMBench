import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from lambench.metrics.metrics_calculations import MetricsCalculations


@pytest.fixture
def metrics_calculations():
    mock_domain_results = MagicMock()
    return MetricsCalculations(mock_domain_results), mock_domain_results


@patch("lambench.metrics.metrics_calculations.CalculatorRecord.query")
def test_fetch_overall_zero_shot_results(mock_query, metrics_calculations):
    metrics, mock_domain_results = metrics_calculations
    model = MagicMock()
    mock_domain_results.aggregate_domain_results_for_one_model.return_value = {
        "task1": 0.8,
        "task2": 0.9,
    }
    result = metrics.fetch_overall_zero_shot_results(model)
    np.testing.assert_almost_equal(result, 0.85)


def test_fetch_generalizability_ood_results(metrics_calculations):
    metrics, mock_domain_results = metrics_calculations
    model1 = MagicMock()
    model1.model_metadata.pretty_name = "Model1"
    model2 = MagicMock()
    model2.model_metadata.pretty_name = "Model2"
    mock_domain_results.leaderboard_models = [model1, model2]

    mock_domain_results.aggregate_domain_results_for_one_model.side_effect = [
        {"task1": 0.8, "task2": 0.9},
        {"task1": 0.7, "task2": 0.6},
    ]

    result = metrics.fetch_generalizability_ood_results()
    expected = {"Model1": 0.15, "Model2": 0.35}
    for key in expected:
        np.testing.assert_almost_equal(result[key], expected[key])


@patch("lambench.metrics.metrics_calculations.CalculatorRecord.query")
def test_fetch_stability_results(mock_query, metrics_calculations):
    metrics, mock_domain_results = metrics_calculations
    model = MagicMock()
    model.model_metadata.pretty_name = "Model1"
    mock_domain_results.leaderboard_models = [model]

    mock_query.return_value = [
        MagicMock(metrics={"std": 0.1, "slope": 0.2, "success_rate": 0.9})
    ]
    with patch(
        "lambench.metrics.metrics_calculations.aggregated_nve_md_results"
    ) as mock_agg:
        mock_agg.return_value = {"std": 0.1, "slope": 0.2, "success_rate": 0.9}
        result = metrics.fetch_stability_results()
        assert "Model1" in result
        np.testing.assert_almost_equal(result["Model1"], 0.8333, decimal=4)


@patch("lambench.metrics.metrics_calculations.CalculatorRecord.query")
def test_fetch_inference_efficiency_results(mock_query, metrics_calculations):
    metrics, _ = metrics_calculations
    model = MagicMock()
    mock_query.return_value = [MagicMock(metrics={"average_time": 0.5})]
    with patch(
        "lambench.metrics.metrics_calculations.aggregated_inference_efficiency_results"
    ) as mock_agg:
        mock_agg.return_value = {"average_time": 0.5}
        result = metrics.fetch_inference_efficiency_results(model)
        assert result == {"average_time": 0.5}


def test_summarize_final_rankings(metrics_calculations):
    metrics, _ = metrics_calculations
    with (
        patch.object(
            metrics, "fetch_generalizability_ood_results"
        ) as mock_generalizability,
        patch.object(metrics, "fetch_applicability_results") as mock_applicability,
    ):
        mock_generalizability.return_value = {"Model1": 0.2, "Model2": 0.4}
        mock_applicability.return_value = {"Model1": 0.8, "Model2": 0.6}
        result = metrics.summarize_final_rankings()
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0]["Model"] == "Model1"
        assert result.iloc[1]["Model"] == "Model2"
