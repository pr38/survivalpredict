import numpy as np


def _get_breslow_base_hazard(
    risk,
    times,
    events,
    max_time,
):
    unique_times = np.arange(1, max_time + 1)
    rows_at_risk_at_time = times[:, np.newaxis] > unique_times

    failure_per_unique_time = np.bincount(
        times.astype(np.int64), events, minlength=max_time + 1
    )[1:]

    risk_per_time = np.dot(risk, rows_at_risk_at_time)
    base_hazard = np.divide(
        failure_per_unique_time,
        risk_per_time,
        out=np.zeros(max_time),
        where=risk_per_time != 0,
    )
    return base_hazard
