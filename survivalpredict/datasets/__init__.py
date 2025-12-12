import os

import numpy as np


def load_iranian_telecom_churn() -> dict[str, np.ndarray]:
    """
    Sample churn data for a 12 months period, from the Iranian telecom company.
    Taken from https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset.

    The dataset was originally used in the following article:
    Jafari-Marandi, R., Denton, J., Idris, A., Smith, B. K., & Keramati, A. (2020). Optimum Profit-Driven Churn Decision Making: Innovative Artificial Neural Networks in Telecom Industry. Neural Computing and Applications.
    """

    dirname = os.path.dirname(os.path.abspath(__file__))

    X = np.loadtxt(os.path.join(dirname, "iranian_churn_X.txt"), delimiter=",")
    times = np.loadtxt(os.path.join(dirname, "iranian_churn_times.txt"), delimiter=",")
    events = np.loadtxt(
        os.path.join(dirname, "iranian_churn_events.txt"), delimiter=","
    )
    col_names = np.loadtxt(
        os.path.join(dirname, "iranian_churn_col_names.txt"), delimiter=",", dtype=str
    )

    return {"X": X, "times": times, "events": events, "column_names": col_names}
