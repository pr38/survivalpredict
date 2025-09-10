import numpy as np


def get_kaplan_meier_survival_curve_from_time_as_int_(
    events: np.ndarray,
    times: np.ndarray,
    max_time: int,
) -> np.ndarray:

    times = times - 1

    death_per_step = np.bincount(times, events, minlength=max_time)
    exit_per_step = np.bincount(times, minlength=max_time)

    right_censor_count_per_step = exit_per_step - death_per_step

    at_risk_per_step = np.flip(
        np.cumsum(np.flip(right_censor_count_per_step + death_per_step))
    )

    hazard_at_step = np.divide(
        death_per_step,
        at_risk_per_step,
        out=np.zeros(max_time).astype(np.float64),
        where=death_per_step != 0,
    )

    return (1 - hazard_at_step).cumprod()
