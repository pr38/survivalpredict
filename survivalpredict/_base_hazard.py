import numpy as np
from typing import Optional


def _get_breslow_base_hazard(
    risk: np.ndarray[tuple[int], np.dtype[np.float64]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    max_time: int,
    times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
):
    unique_times = np.arange(1, max_time + 1)
    rows_at_risk_at_time = times[:, np.newaxis] > unique_times

    failure_per_unique_time = np.bincount(
        times.astype(np.int64), events, minlength=max_time + 1
    )[1:]

    if times_start is not None:
        rows_at_risk_at_time = np.logical_and(
            rows_at_risk_at_time, times_start[:, np.newaxis] < unique_times
        )

    risk_per_time = np.dot(risk, rows_at_risk_at_time)
    base_hazard = np.divide(
        failure_per_unique_time,
        risk_per_time,
        out=np.zeros(max_time),
        where=risk_per_time != 0,
    )
    return base_hazard
