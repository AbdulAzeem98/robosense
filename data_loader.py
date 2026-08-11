"""
RoboSense: Data loading utilities for the UCI Robot Execution Failures dataset.

Each .data.txt file contains blocks separated by blank lines:
    <class_label>
    <F x> <F y> <F z> <T x> <T y> <T z>   (15 rows, one per timestep)
"""
import numpy as np
import pandas as pd


def load_lp_file(path: str):
    """Parse one lpN.data.txt file into (X, y).

    Returns
    -------
    X : np.ndarray, shape (n_instances, 15, 6)
        Raw force/torque time series per instance.
    y : np.ndarray, shape (n_instances,)
        Class label string per instance.
    """
    instances, labels = [], []
    current_rows = []
    current_label = None

    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    for line in lines:
        if line.strip() == "":
            continue
        if not line.startswith("\t") and not line.startswith(" "):
            # This is a class-label line -> flush previous instance
            if current_label is not None and len(current_rows) == 15:
                instances.append(current_rows)
                labels.append(current_label)
            current_label = line.strip()
            current_rows = []
        else:
            values = [int(v) for v in line.strip().split()]
            current_rows.append(values)

    # flush last instance
    if current_label is not None and len(current_rows) == 15:
        instances.append(current_rows)
        labels.append(current_label)

    X = np.array(instances, dtype=float)   # (n, 15, 6)
    y = np.array(labels)
    return X, y


def load_dataset(path: str = "data/lp1.data.txt"):
    X, y = load_lp_file(path)
    print(f"Loaded {path}: X={X.shape}, classes={sorted(set(y))}")
    for cls in sorted(set(y)):
        print(f"  {cls:20s}: {sum(y == cls)}")
    return X, y


if __name__ == "__main__":
    load_dataset("data/lp1.data.txt")
