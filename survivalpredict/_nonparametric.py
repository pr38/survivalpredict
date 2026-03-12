import numba as nb
import numpy as np

_get_kaplan_meier_survival_curve_signature = nb.types.Array(
    nb.types.float64, 1, "C", False, aligned=True
)(
    nb.types.Array(nb.types.boolean, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.int64,
)


@nb.njit(_get_kaplan_meier_survival_curve_signature, cache=True)
def get_kaplan_meier_survival_curve(
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    max_time: int,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

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


get_kaplan_meier_survival_curve_with_left_censorship_signature_ = nb.types.Array(
    nb.types.float64, 1, "C", False, aligned=True
)(
    nb.types.Array(nb.types.boolean, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.int64,
)


@nb.njit(get_kaplan_meier_survival_curve_with_left_censorship_signature_, cache=True)
def get_kaplan_meier_survival_curve_with_left_censorship(
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    times_start: np.ndarray[tuple[int], np.dtype[np.int64]],
    max_time: int,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    bin_length = times.max() + 1

    death_per_step = np.bincount(times, events, minlength=bin_length)
    censor_at_step = np.bincount(times, np.logical_not(events), minlength=bin_length)
    enter_at_step = np.bincount(times_start, minlength=bin_length)

    at_risk_per_step = np.flip(
        np.cumsum(np.flip(censor_at_step + death_per_step - enter_at_step))
    )

    hazard_at_step = death_per_step / at_risk_per_step

    hazard_at_step = np.where(death_per_step != 0, hazard_at_step, 0)
    survival_curve = np.cumprod(1 - hazard_at_step)

    if len(survival_curve) > max_time + 1:
        survival_curve = survival_curve[: max_time + 1]
    elif len(survival_curve) < max_time + 1:
        missing_dims = max_time - len(survival_curve)

        impulted_values = np.repeat(survival_curve[-1], missing_dims + 1)

        survival_curve = np.hstack((survival_curve, impulted_values))

    return survival_curve[1:]  # exclude time 0
