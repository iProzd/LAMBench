from lambench.models.ase_models import ASEModel
from ase.io import read
from ase.atoms import Atoms
import logging
import time
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="infer.log",
)


def get_efv(atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray]:
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    stress = atoms.get_stress()
    v = (
        -np.array(
            [
                [stress[0], stress[5], stress[4]],
                [stress[5], stress[1], stress[3]],
                [stress[4], stress[3], stress[2]],
            ]
        )
        * atoms.get_volume()
    )
    return e, f, v


def run_inference(
    model: ASEModel, test_data: Path, warmup_ratio: float
) -> dict[str, dict[str, float]]:
    """
    Inference for all systems, return average time and success rate for each system.
    """
    results = {}
    subfolders = [subfolder for subfolder in test_data.iterdir() if subfolder.is_dir()]
    for subfolder in subfolders:
        system_name = subfolder.name
        try:
            system_result = run_one_inference(model, subfolder, warmup_ratio)
            average_time = system_result["average_time_per_step"]
            success_rate = system_result["success_rate"]
            results[system_name] = {
                "average_time_per_step": average_time,
                "success_rate": success_rate,
            }
            logging.info(
                f"Inference completed for system {system_name} with average time {average_time} s and success rate {success_rate:.2f}%"
            )
        except Exception as e:
            logging.error(f"Error in inference for system {system_name}: {e}")
            results[system_name] = {"average_time_per_step": None, "success_rate": 0.0}
    return results


def run_one_inference(
    model: ASEModel, test_data: Path, warmup_ratio: float
) -> dict[str, float]:
    """
    Infer for one system, return averaged time and success rate, starting timing at warmup_ratio.
    """
    test_files = list(test_data.glob("*.vasp"))
    test_atoms = [read(file) for file in test_files]
    start_index = int(len(test_atoms) * warmup_ratio)
    total_time = 0
    valid_steps = 0
    successful_inferences = 0
    total_inferences = len(test_atoms)

    for i, atoms in enumerate(test_atoms):
        atoms.calc = model.calc
        start = time.time()
        try:
            get_efv(atoms)
            successful_inferences += 1
        except Exception as e:
            logging.error(f"Error in inference for {str(atoms.symbols)}: {e}")
            continue

        end = time.time()
        elapsed_time = end - start

        if i >= start_index:
            total_time += elapsed_time
            valid_steps += 1

        logging.info(
            f"Inference completed for system {str(atoms.symbols)} in {elapsed_time} s"
        )

    if valid_steps > 0:
        average_time_per_step = total_time / valid_steps
    else:
        average_time_per_step = np.nan

    if total_inferences > 0:
        success_rate = (successful_inferences / total_inferences) * 100
    else:
        success_rate = 0.0

    return {
        "average_time_per_step": average_time_per_step,
        "success_rate": success_rate,
    }
