import numpy as np


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
