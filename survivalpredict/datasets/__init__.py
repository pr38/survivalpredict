import os

import numpy as np

__all__ = [
    "load_iranian_telecom_churn",
    "load_kickstarter",
]


def load_iranian_telecom_churn() -> dict[str, np.ndarray]:
    """
    Load iranian telecom churn datset.

    Sample churn data for a 12 months period, from the Iranian telecom company. Taken
    from https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset.

    The dataset was originally used in the following article: Jafari-Marandi, R.,
    Denton, J., Idris, A., Smith, B. K., & Keramati, A. (2020). Optimum Profit-Driven
    Churn Decision Making: Innovative Artificial Neural Networks in Telecom Industry.
    Neural Computing and Applications.

    Returns
    ----------

    dict of arrays

        A dict of arrays containing iranian telecom churn datset components. The
        possible keys for this ``dict`` are:

        ``X``
            The data matrix.

        ``times``
            Associated encoding for last known interval of survival.

        ``events``
            '1' if churned, '0' is censored.

        ``column_names``
            Column names of the 'X' array.
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


def load_kickstarter() -> dict[str, np.ndarray]:
    """
    Load kickstarter datset.

    This dataset was originally hosted by by professor Chandan Reddy's virginia tech
    site. taken from https://dmkd.cs.vt.edu/projects/survival/data/

    Returns
    ----------

    dict of arrays

        A dict of arrays containing kickstarter datset components. The possible keys for
        this ``dict`` are:

        ``X``
            The data matrix.

        ``times``
            Associated encoding for last known interval of survival.

        ``events``
            '1' if churned, '0' is censored.

        ``column_names``
            Column names of the 'X' array.
    """
    dirname = os.path.dirname(os.path.abspath(__file__))

    X = np.loadtxt(os.path.join(dirname, "kickstarter_X.txt"), delimiter=",")
    times = np.loadtxt(os.path.join(dirname, "kickstarter_times.txt"), delimiter=",")
    events = np.loadtxt(os.path.join(dirname, "kickstarter_events.txt"), delimiter=",")
    col_names = np.loadtxt(
        os.path.join(dirname, "kickstarter_col_names.txt"), delimiter=",", dtype=str
    )

    return {"X": X, "times": times, "events": events, "column_names": col_names}
