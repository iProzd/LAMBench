import os
from pathlib import Path

def prepare_dptest_input_file():
    raise NotImplementedError

def prepare_finetune_input_file():
    raise NotImplementedError

def parse_dptest_log_file(filepath:Path, txt_type:str="standard") -> dict[str,float]:
    """
    Parse dptest results to a dict

    Parameters:
    ----------
        filepath: str
            The path to the dptest output logfile.
        txt_type: str
            The type of dptest output file. Options are "standard" and "property".

    """
    with open(filepath,"r") as f:
        content = f.readlines()

    if txt_type == "standard":
        metrics = {}
        for line in content[-11:-1]:
            line = line.split("deepmd.entrypoints.test")[-1].strip()
            # Force  MAE/natoms -> force_mae_natoms in compliance with DirectPredictRecord
            metrics[
                line.split(":")[0]
                .strip()
                .replace("  ", " ")
                .replace(" ", "_")
                .replace("/", "_")
                .lower()
            ] = float(line.split(":")[-1].strip().split(" ")[0])
    elif txt_type == "property":
        metrics = {}
        metrics["PROPERTY MAE"] = float(content[-3].split(":")[-1].split("units")[0].strip())
        metrics["PROPERTY RMSE"] = float(content[-2].split(":")[-1].split("units")[0].strip())
    else:
        raise ValueError(f"Unknown dptest output type: {txt_type}")
    return metrics