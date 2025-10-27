import numba as nb
import numpy as np

get_kaplan_meier_survival_curve_from_time_as_int_signature_ = nb.types.Array(
    nb.types.float64, 1, "C", False, aligned=True
)(
    nb.types.Array(nb.types.boolean, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.int64,
)


@nb.njit(get_kaplan_meier_survival_curve_from_time_as_int_signature_)
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
    )
    hazard_at_step = np.where(death_per_step != 0, hazard_at_step, 0)

    return (1 - hazard_at_step).cumprod()
