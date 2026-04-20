import numba as nb
import numpy as np
from scipy.optimize import minimize


@nb.njit(cache=True)
def at_risk_per_time_with_start_times(
    p_exp, time_end_return_inverse, time_start_return_inverse, n_unique_times
):

    n_rows = len(p_exp)
    output = np.zeros(n_unique_times)

    for i in range(n_rows):
        output[time_start_return_inverse[i] : time_end_return_inverse[i]] += p_exp[i]

    return output


@nb.njit(cache=True)
def bincount_reverse_cumsum_along_axis(
    XxXb, time_end_return_inverse, time_start_return_inverse, n_unique_times
):
    output = np.zeros((n_unique_times, XxXb.shape[1]))
    n_rows = XxXb.shape[0]

    for i in range(n_rows):
        output[time_start_return_inverse[i] : time_end_return_inverse[i]] += XxXb[i]

    return output


@nb.njit(cache=True)
def get_breslow_n_log_likeliehood_with_left_censorship(
    weights,
    X,
    events,
    time_end_return_inverse,
    time_start_return_inverse,
    event_counts_at_times,
    n_unique_times,
):
    p = np.dot(X, weights)
    p_exp = np.exp(p)
    risk_set_at_time = at_risk_per_time_with_start_times(
        p_exp, time_end_return_inverse, time_start_return_inverse, n_unique_times
    )
    risk_set = risk_set_at_time[time_end_return_inverse - 1]
    return -np.sum(np.nan_to_num(events * (p - np.log(risk_set))))


get_breslow_n_log_likeliehood_with_left_censorship_with_strata_sig = nb.types.float64(
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
    nb.types.List(nb.types.Array(nb.types.float64, 2, "C"), True),
    nb.types.List(nb.types.Array(nb.types.bool_, 1, "C"), True),
    nb.types.List(nb.types.int64, True),
    nb.types.List(nb.types.Array(nb.types.float64, 1, "C"), True),
    nb.types.List(nb.types.Array(nb.types.int64, 1, "C"), True),
    nb.types.List(nb.types.Array(nb.types.int64, 1, "C"), True),
    nb.types.int64,
    nb.types.float64,
    nb.types.float64,
)


@nb.njit(get_breslow_n_log_likeliehood_with_left_censorship_with_strata_sig, cache=True)
def get_breslow_n_log_likeliehood_with_left_censorship_with_strata(
    weights,
    X_strata,
    events_strata,
    n_unique_times_strata,
    event_counts_at_times_strata,
    time_return_inverse_strata,
    time_start_return_inverse_strata,
    n_strata,
    alpha,
    l1_ratio,
):

    l1 = alpha * l1_ratio * np.abs(weights).sum()
    l2 = 0.5 * alpha * (1.0 - l1_ratio) * np.square(weights).sum()
    loss = l1 + l2

    for i in range(n_strata):
        loss += get_breslow_n_log_likeliehood_with_left_censorship(
            weights,
            X_strata[i],
            events_strata[i],
            time_return_inverse_strata[i],
            time_start_return_inverse_strata[i],
            event_counts_at_times_strata[i],
            n_unique_times_strata[i],
        )

    return loss


@nb.njit(cache=True)
def get_breslow_jacobian_with_left_censorship(
    weights,
    X,
    events,
    time_end_return_inverse,
    time_start_return_inverse,
    event_counts_at_times,
    n_unique_times,
):

    p = np.dot(X, weights)
    p_exp = np.exp(p)
    risk_set_at_time = at_risk_per_time_with_start_times(
        p_exp, time_end_return_inverse, time_start_return_inverse, n_unique_times
    )
    risk_set = risk_set_at_time[time_end_return_inverse - 1]

    XxXb = np.multiply(X, p_exp[:, np.newaxis])

    XxXb_at_Xt_at_time_cumsum = bincount_reverse_cumsum_along_axis(
        XxXb, time_end_return_inverse, time_start_return_inverse, n_unique_times
    )

    XxXb_at_Xt_at_index = XxXb_at_Xt_at_time_cumsum[time_end_return_inverse - 1]

    XxXb_at_Xt_at_index_div_riskset = XxXb_at_Xt_at_index / risk_set[:, np.newaxis]

    jacobian_presum = events[:, np.newaxis] * (X - XxXb_at_Xt_at_index_div_riskset)

    jacobian_presum = np.nan_to_num(jacobian_presum)

    jacobian = -np.sum(jacobian_presum, axis=0)

    return jacobian


get_breslow_jacobian_with_left_censorship_with_strata_sig = nb.types.Array(
    nb.types.float64, 1, "C"
)(
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
    nb.types.List(nb.types.Array(nb.types.float64, 2, "C"), True),
    nb.types.List(nb.types.Array(nb.types.bool_, 1, "C"), True),
    nb.types.List(nb.types.int64, True),
    nb.types.List(nb.types.Array(nb.types.float64, 1, "C"), True),
    nb.types.List(nb.types.Array(nb.types.int64, 1, "C"), True),
    nb.types.List(nb.types.Array(nb.types.int64, 1, "C"), True),
    nb.types.int64,
    nb.types.float64,
    nb.types.float64,
)


@nb.njit(get_breslow_jacobian_with_left_censorship_with_strata_sig, cache=True)
def get_breslow_jacobian_with_left_censorship_with_strata(
    weights,
    X_strata,
    events_strata,
    n_unique_times_strata,
    event_counts_at_times_strata,
    time_return_inverse_strata,
    time_start_return_inverse_strata,
    n_strata,
    alpha,
    l1_ratio,
):
    l1_jacobian = np.sign(weights) * alpha * l1_ratio
    l2_jacobian = 2 * weights * (0.5 * alpha) * (1 - l1_ratio)

    jacobian = l1_jacobian + l2_jacobian

    for i in range(n_strata):
        jacobian += get_breslow_jacobian_with_left_censorship(
            weights,
            X_strata[i],
            events_strata[i],
            time_return_inverse_strata[i],
            time_start_return_inverse_strata[i],
            event_counts_at_times_strata[i],
            n_unique_times_strata[i],
        )

    return jacobian


def train_cox_ph_breslow_with_left_censorship_scipy_minimize(
    X_strata,
    events_strata,
    n_unique_times_strata,
    event_counts_at_times_strata,
    time_return_inverse_strata,
    time_start_return_inverse_strata,
    n_strata,
    alpha,
    l1_ratio,
    weights,
    tol,
    method,
):
    result = minimize(
        get_breslow_n_log_likeliehood_with_left_censorship_with_strata,
        weights,
        jac=get_breslow_jacobian_with_left_censorship_with_strata,
        args=(
            X_strata,
            events_strata,
            n_unique_times_strata,
            event_counts_at_times_strata,
            time_return_inverse_strata,
            time_start_return_inverse_strata,
            n_strata,
            alpha,
            l1_ratio,
        ),
        tol=tol,
        method=method,
    )

    weights = result.x
    loss = result.fun

    return weights, loss
