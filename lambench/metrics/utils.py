import numpy as np


def aggregated_nve_md_results(results: list[dict[str, float]]) -> dict[str, float]:
    # Aggregate results
    aggregated_result = {
        "simulation_time": np.mean(
            [
                result["simulation_time"]
                if result["simulation_time"] is not None
                else np.nan
                for result in results
            ]
        ),
        "steps": np.mean(
            [result["steps"] if result["steps"] != 0 else np.nan for result in results]
        ),
        "slope": log_average([result["slope"] for result in results]),
        "momenta_diff": log_average([result["momenta_diff"] for result in results]),
    }
    return aggregated_result


def calculate_nve_md_score(
    aggregated_result, division_protection: float = 1e-6
) -> dict[str, float]:
    """
    This function aggreate the results across all four metrics and return the final result.
    """
    final_result = np.log(
        aggregated_result["steps"]
        / (
            aggregated_result["simulation_time"]
            * (aggregated_result["energy_std"] + division_protection)
            * (np.abs(aggregated_result["slope"]) + division_protection)
        )
    )
    return {"NVE Score": final_result}


def log_average(resutls: list[float]) -> float:
    """
    A function to calculate the log average of a list of results to avoid overwheelmingly large numbers.
    """
    if not resutls:
        return np.nan
    elif any(np.isnan(resutls)):
        return np.nan

    return np.exp(np.mean(np.log(resutls)))
