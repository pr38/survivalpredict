import numpy as np


def _get_breslow_base_hazard(
    X,
    times,
    events,
    max_time,
    coef,
):
    max_time = int(max(times))
    unique_times = np.arange(1, max_time + 1)
    rows_at_risk_at_time = times[:, np.newaxis] > unique_times

    failure_per_unique_time = np.bincount(
        times.astype(np.int64), events, minlength=max_time
    )[1:]

    risk = np.exp(np.dot(X, coef))
    risk_per_time = np.dot(risk, rows_at_risk_at_time)
    base_hazard = np.divide(
        failure_per_unique_time,
        risk_per_time,
        out=np.zeros(max_time),
        where=risk_per_time != 0,
    )
    return base_hazard


def _estiamte_chen_base_hazard():
    pass


def _estiamte_weibull_base_hazard():
    pass
