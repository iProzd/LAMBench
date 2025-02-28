from lambench.models.ase_models import ASEModel
from ase import Atoms
from ase.io import read
import logging
import time
import numpy as np
from typing import List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w', filename='infer.log')


def run_batch_infer(
    model: ASEModel,
    test_data: Path
) -> Dict[str, float]:
    """
    Infer for all batches
    """
    results = {}
    subfolders = [subfolder for subfolder in test_data.iterdir() if subfolder.is_dir()]
    for subfolder in subfolders:
        system_name = subfolder.name
        try:
            batch_result = run_one_batch_infer(model, subfolder)
            average_time = batch_result["average_time_per_step"]
            results[system_name] = average_time
            logging.info(f"Batch inference completed for system {system_name} with average time {average_time} s")
        except Exception as e:
            logging.error(f"Error in batch inference for system {system_name}: {e}")
    return results


def run_one_batch_infer(
    model: ASEModel,
    test_data: Path
) -> Dict[str, float]:
    """
    Infer for one batch, return averaged time, starting timing at 20%.
    """
    test_files = list(test_data.glob("*.vasp"))
    test_atoms = [read(file) for file in test_files]
    start_index = int(len(test_atoms) * 0.2)
    total_time = 0
    valid_steps = 0
    for i, atoms in enumerate(test_atoms):
        atoms.calc = model.calc
        start = time.time()
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            volume = atoms.get_volume()
            stress_tensor = np.zeros((3, 3))
            stress_tensor[0, 0] = stress[0]
            stress_tensor[1, 1] = stress[1]
            stress_tensor[2, 2] = stress[2]
            stress_tensor[1, 2] = stress[3]
            stress_tensor[0, 2] = stress[4]
            stress_tensor[0, 1] = stress[5]
            stress_tensor[2, 1] = stress[3]
            stress_tensor[2, 0] = stress[4]
            stress_tensor[1, 0] = stress[5]
            virial = -stress_tensor * volume
        except Exception as e:
            logging.error(f"Error in inference for {str(atoms.symbols)}: {e}")
            continue

        end = time.time()
        elapsed_time = end - start

        if i >= start_index:
            total_time += elapsed_time
            valid_steps += 1

        logging.info(f"Inference completed for system {str(atoms.symbols)} in {elapsed_time} s")

    if valid_steps > 0:
        average_time_per_step = total_time / valid_steps
    else:
        average_time_per_step = np.nan

    return {
        "average_time_per_step": average_time_per_step,
    }