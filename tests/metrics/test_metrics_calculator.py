import pytest
from unittest.mock import MagicMock
import numpy as np
from lambench.metrics.vishelper.metrics_calculations import MetricsCalculator


@pytest.fixture
def mock_raw_results():
    return MagicMock()


@pytest.fixture
def metrics_calculator(mock_raw_results):
    return MetricsCalculator(mock_raw_results)


def test_calculate_mean_m_bar_domain(metrics_calculator, mock_raw_results):
    model = MagicMock()
    mock_raw_results.aggregate_ood_results_for_one_model.return_value = {
        "domain1": 0.8,
        "domain2": 0.9,
    }
    result = metrics_calculator.calculate_mean_m_bar_domain(model)
    np.testing.assert_almost_equal(result, 0.85)


def test_convert_metric_to_score_minmax(metrics_calculator):
    metric_dict = {"model1": 0.2, "model2": 0.8, "model3": 0.5}
    result = metrics_calculator.convert_metric_to_score(metric_dict, method="minmax")
    expected = {"model1": 1.0, "model2": 0.0, "model3": 0.5}
    for key in result:
        np.testing.assert_almost_equal(result[key], expected[key])


def test_convert_metric_to_score_log(metrics_calculator):
    metric_dict = {"model1": 0.2, "model2": 0.8, "model3": 0.5}
    result = metrics_calculator.convert_metric_to_score(metric_dict, method="-log")
    assert result["model1"] > result["model3"] > result["model2"]


def test_calculate_generalizability_ood_score(metrics_calculator, mock_raw_results):
    mock_raw_results.aggregate_ood_results.return_value = {
        "model1": {"domain1": 0.8, "domain2": 0.9},
        "model2": {"domain1": 0.7, "domain2": 0.6},
    }
    result = metrics_calculator.calculate_generalizability_ood_score()
    np.testing.assert_almost_equal(result["model2"], 1)
    np.testing.assert_almost_equal(result["model1"], 0.41594, decimal=5)


def test_calculate_stability_results(metrics_calculator, mock_raw_results):
    mock_raw_results.fetch_stability_results.return_value = {
        "model1": {"std": 0.1, "slope": 0.2, "success_rate": 0.9},
        "model2": {"std": 0.3, "slope": 0.1, "success_rate": 0.8},
    }
    result = metrics_calculator.calculate_stability_results()
    assert "model1" in result and "model2" in result
    assert result["model1"] > result["model2"]


def test_calculate_efficiency_results(metrics_calculator, mock_raw_results):
    mock_raw_results.fetch_inference_efficiency_results.return_value = {
        "model1": {"average_time": 0.5},
        "model2": {"average_time": 0.8},
    }
    result = metrics_calculator.calculate_efficiency_results()
    np.testing.assert_almost_equal(result["model1"], 1)
    np.testing.assert_almost_equal(result["model2"], 0)


def test_calculate_applicability_results(metrics_calculator, mock_raw_results):
    metrics_calculator.calculate_efficiency_results = MagicMock(
        return_value={"model1": 0.9, "model2": 0.7}
    )
    metrics_calculator.calculate_stability_results = MagicMock(
        return_value={"model1": 0.8, "model2": 0.6}
    )
    result = metrics_calculator.calculate_applicability_results()
    np.testing.assert_almost_equal(result["model1"], 0.85)
    np.testing.assert_almost_equal(result["model2"], 0.65)


def test_summarize_final_rankings(metrics_calculator):
    metrics_calculator.calculate_generalizability_ood_score = MagicMock(
        return_value={"model1": 0.8, "model2": 0.6}
    )
    metrics_calculator.calculate_applicability_results = MagicMock(
        return_value={"model1": 0.9, "model2": 0.7}
    )
    result = metrics_calculator.summarize_final_rankings()
    assert result is not None
    assert result.iloc[0]["Model"] == "model1"
    assert result.iloc[1]["Model"] == "model2"
