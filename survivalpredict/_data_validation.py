import numpy as np


def _as_numeric_np_array(a):
    try:
        return np.array(a).astype(np.float64, order="C")
    except:
        raise ValueError("the feature array must be numeric")


def _as_int_np_array(a, param_name="times array"):
    try:
        a = np.array(a)
        if np.issubdtype(np.dtype, np.integer):
            dtype = a.dtype
        else:
            dtype = np.int64

        a = a.astype(dtype)
        return a
    except:
        raise ValueError(f"{param_name} must be array-like with integer data type")


def _as_bool_np_array(a):
    try:
        a = np.array(a)
        a = a.astype(np.bool_)
        return a
    except:
        raise ValueError("the event array must be bool type")


def _as_int(a, param_name):
    try:
        return int(a)
    except ValueError as e:
        raise ValueError(f"{param_name} should be integer")


def validate_times_array(a):
    a = _as_int_np_array(a)
    if a.min() < 1:
        raise ValueError(
            "for the times array it is assumed that event and right censorship events to start at 1"
        )
    return a


def validate_times_start_array(times_start, times):
    times_start = _as_int_np_array(times_start, "times_start array")
    if times_start.min() < 0:
        raise ValueError("times_start array assumes the lowest value is 0")
    if (times_start >= times).all():
        raise ValueError("times_start array should be less than times array")

    return times_start


def validate_survival_data(X, times, events):
    X = _as_numeric_np_array(X)
    times = validate_times_array(times)
    events = _as_bool_np_array(events)
    return X, times, events
