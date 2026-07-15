import numba as nb
import numpy as np

_get_kaplan_meier_survival_curve_signature = nb.types.Array(
    nb.types.float64, 1, "C", False, aligned=True
)(
    nb.types.Array(nb.types.boolean, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.int64,
    nb.types.boolean,
)


@nb.njit(_get_kaplan_meier_survival_curve_signature, cache=True)
def get_kaplan_meier_survival_curve(
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    max_time: int,
    return_hazard: bool = False,
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

    if return_hazard:
        output = hazard_at_step
    else: #return survival
        output = (1 - hazard_at_step).cumprod()


    if len(output) > max_time:
        output = output[:max_time]
    elif len(output) < max_time:
        missing_dims = max_time - len(output)

        if return_hazard:
            impulted_values = np.repeat(0.0, missing_dims)

        else: #return survival
            impulted_values = np.repeat(output[-1], missing_dims)
        

        output = np.hstack((output, impulted_values))

    return output


get_kaplan_meier_survival_curve_with_left_censorship_signature_ = nb.types.Array(
    nb.types.float64, 1, "C", False, aligned=True
)(
    nb.types.Array(nb.types.boolean, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.int64,
    nb.types.boolean,
)


@nb.njit(get_kaplan_meier_survival_curve_with_left_censorship_signature_, cache=True)
def get_kaplan_meier_survival_curve_with_left_censorship(
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    times_start: np.ndarray[tuple[int], np.dtype[np.int64]],
    max_time: int,
    return_hazard: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    bin_length = times.max() + 1

    death_per_step = np.bincount(times, events, minlength=bin_length)
    censor_at_step = np.bincount(times, np.logical_not(events), minlength=bin_length)
    enter_at_step = np.bincount(times_start, minlength=bin_length)

    at_risk_per_step = np.flip(
        np.cumsum(np.flip(censor_at_step + death_per_step - enter_at_step))
    )

    hazard_at_step = death_per_step / at_risk_per_step

    if return_hazard:
        output = hazard_at_step
    else: #return survival
        hazard_at_step = np.where(death_per_step != 0, hazard_at_step, 0)
        output = np.cumprod(1 - hazard_at_step) #survival_curve


    if len(output) > max_time + 1:
        output = output[: max_time + 1]
    elif len(output) < max_time + 1:
        missing_dims = max_time - len(output)

        if return_hazard:
            impulted_values = np.repeat(0.0, missing_dims + 1)
        else: #return survival
            impulted_values = np.repeat(output[-1], missing_dims + 1)

        output = np.hstack((output, impulted_values))

    return output[1:]  # exclude time 0
