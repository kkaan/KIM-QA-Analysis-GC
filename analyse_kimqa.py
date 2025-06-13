import numpy as np
import pandas as pd
from typing import Tuple


def load_coordinates(file_path: str) -> np.ndarray:
    """Read marker and isocenter coordinates from a text file."""
    data = pd.read_csv(file_path, sep="\s+", header=None, comment="#")
    return data.values


def analyse_static(kim_data: np.ndarray, motion_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean, std and percentiles for static tests."""
    diff = kim_data - motion_data
    mean = diff.mean(axis=0)
    std = diff.std(axis=0, ddof=0)
    pct = np.percentile(diff, [5, 95], axis=0)
    return mean, std, pct


def main():
    """Example analysis workflow."""
    # Example file paths (replace with real ones)
    coord_file = "Test Files/Varian/co-ords.txt"
    motion_file = "Robot traces/LiverTraj_LargeSIandAP70s_robot.txt"

    # Load coordinate and motion data
    coords = load_coordinates(coord_file)
    motion = pd.read_csv(motion_file, sep="\s+", header=None).values

    # Use first three columns of motion data (LR, SI, AP)
    motion = motion[:, :3]

    mean, std, pct = analyse_static(coords[:-1, :], motion[:coords.shape[0]-1, :])

    print("Mean:", mean)
    print("Std:", std)
    print("Percentiles:", pct)


if __name__ == "__main__":
    main()
