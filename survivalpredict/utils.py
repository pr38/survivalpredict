import numpy as np


def _as_numeric_np_array(a):
    try:
        a = np.array(a)
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(np.float64, np.integer):
            dtype = a.dtype
        else:
            dtype = np.float64

        a = a.astype(dtype)
        return a
    except:
        raise ValueError("the feature array must be numeric")


def _as_int_np_array(a):
    try:
        a = np.array(a)
        if np.issubdtype(np.dtype, np.integer):
            dtype = a.dtype
        else:
            dtype = np.int64

        a = a.astype(dtype)
        return a
    except:
        raise ValueError("the times array must be an integer type")


def _as_bool_np_array(a):
    try:
        a = np.array(a)
        a = a.astype(np.bool_)
        return a
    except:
        raise ValueError("the event array must be an bool type")


def _as_int(a, param_name):
    try:
        return int(a)
    except ValueError as e:
        raise ValueError(f"{param_name} should be an integer")


def validate_survival_data(X, times, events):
    X = _as_numeric_np_array(X)
    times = _as_int_np_array(times)
    events = _as_bool_np_array(events)
    return X, times, events
