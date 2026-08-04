import numba as nb
import numpy as np


@nb.njit(
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True)(
        nb.types.Array(nb.types.boolean, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.int64,
        nb.types.boolean,
    ),
    cache=True,
)
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
    else:  # return survival
        output = (1 - hazard_at_step).cumprod()

    if len(output) > max_time:
        output = output[:max_time]
    elif len(output) < max_time:
        missing_dims = max_time - len(output)

        if return_hazard:
            impulted_values = np.repeat(0.0, missing_dims)

        else:  # return survival
            impulted_values = np.repeat(output[-1], missing_dims)

        output = np.hstack((output, impulted_values))

    return output


@nb.njit(
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True)(
        nb.types.Array(nb.types.boolean, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.int64,
        nb.types.boolean,
    ),
    cache=True,
)
def get_kaplan_meier_survival_curve_with_weights(
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    weights: np.ndarray[tuple[int], np.dtype[np.int64]],
    max_time: int,
    return_hazard: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    times = times - 1

    death_per_step = np.bincount(times, events * weights, minlength=max_time)
    exit_per_step = np.bincount(times, weights, minlength=max_time)

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
    else:  # return survival
        output = (1 - hazard_at_step).cumprod()

    if len(output) > max_time:
        output = output[:max_time]
    elif len(output) < max_time:
        missing_dims = max_time - len(output)

        if return_hazard:
            impulted_values = np.repeat(0.0, missing_dims)

        else:  # return survival
            impulted_values = np.repeat(output[-1], missing_dims)

        output = np.hstack((output, impulted_values))

    return output


@nb.njit(
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True)(
        nb.types.Array(nb.types.boolean, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.int64,
        nb.types.boolean,
    ),
    cache=True,
)
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
    else:  # return survival
        hazard_at_step = np.where(death_per_step != 0, hazard_at_step, 0)
        output = np.cumprod(1 - hazard_at_step)  # survival_curve

    if len(output) > max_time + 1:
        output = output[: max_time + 1]
    elif len(output) < max_time + 1:
        missing_dims = max_time - len(output)

        if return_hazard:
            impulted_values = np.repeat(0.0, missing_dims + 1)
        else:  # return survival
            impulted_values = np.repeat(output[-1], missing_dims + 1)

        output = np.hstack((output, impulted_values))

    return output[1:]  # exclude time 0


@nb.njit(
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True)(
        nb.types.Array(nb.types.boolean, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.int64,
        nb.types.boolean,
    ),
    cache=True,
)
def get_kaplan_meier_survival_curve_with_weights_and_left_censorship(
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    times_start: np.ndarray[tuple[int], np.dtype[np.int64]],
    weights: np.ndarray[tuple[int], np.dtype[np.float64]],
    max_time: int,
    return_hazard: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    bin_length = times.max() + 1

    death_per_step = np.bincount(times, events * weights, minlength=bin_length)
    censor_at_step = np.bincount(
        times, np.logical_not(events).astype(np.float64) * weights, minlength=bin_length
    )
    enter_at_step = np.bincount(times_start, weights, minlength=bin_length)

    at_risk_per_step = np.flip(
        np.cumsum(np.flip(censor_at_step + death_per_step - enter_at_step))
    )

    hazard_at_step = death_per_step / at_risk_per_step

    if return_hazard:
        output = hazard_at_step
    else:  # return survival
        hazard_at_step = np.where(death_per_step != 0, hazard_at_step, 0)
        output = np.cumprod(1 - hazard_at_step)  # survival_curve

    if len(output) > max_time + 1:
        output = output[: max_time + 1]
    elif len(output) < max_time + 1:
        missing_dims = max_time - len(output)

        if return_hazard:
            impulted_values = np.repeat(0.0, missing_dims + 1)
        else:  # return survival
            impulted_values = np.repeat(output[-1], missing_dims + 1)

        output = np.hstack((output, impulted_values))

    return output[1:]  # exclude time 0


@nb.njit(
    nb.types.Array(nb.types.float64, 2, "A", False, aligned=True)(
        nb.types.Array(nb.types.float64, 2, "A", False, aligned=True)
    )
)
def convert_kaplan_meier_survival_curve_to_hazards(
    survival: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    one_plus_hazards = np.empty_like(survival)

    for row_index, row_survival in enumerate(survival):
        previous_value = 1.0
        is_not_dead = True
        for col_index, value in enumerate(row_survival):
            if is_not_dead:
                one_plus_hazards[row_index, col_index] = value / previous_value
                if value == 0.0:
                    is_not_dead = False
            else:
                one_plus_hazards[row_index, col_index] = 1.0
            previous_value = value

    return 1 - one_plus_hazards
