import numba as nb
import numpy as np
from scipy.optimize import minimize

from ._cox_ph_estimation import elasticnet_loss_jacobian_hessian

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


self_outterproduct_mul_groupby_time_sig = nb.types.Array(
    nb.types.float64, 3, "C", False, aligned=True
)(
    nb.types.Array(nb.types.float64, 2, "A", False, aligned=True),
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.int64,
)


@nb.jit(self_outterproduct_mul_groupby_time_sig, cache=True)
def self_outterproduct_mul_groupby_broadcast_to_at_risk(
    X, p_exp, time_end_return_inverse, time_start_return_inverse, max_time_index
):

    n_rows = X.shape[0]
    n_col = X.shape[1]

    c = np.empty((n_col, n_col))
    output = np.zeros((max_time_index, n_col, n_col))

    for i in range(n_rows):
        X_i = X[i]
        p_i = p_exp[i]

        for e in range(n_col):
            for f in range(n_col):
                c[e, f] = X_i[e] * X_i[f] * p_i
        output[time_start_return_inverse[i] : time_end_return_inverse[i]] += c

    return output


def breslow_neg_log_likelihood_loss_jacobian_hessian_left_censorship(
    weights,
    X,
    events,
    time_end_return_inverse,
    time_start_return_inverse,
    n_unique_times,
    event_counts_at_times,
):
    p = np.dot(X, weights)
    p_exp = np.exp(p)
    risk_set_at_time = at_risk_per_time_with_start_times(
        p_exp, time_end_return_inverse, time_start_return_inverse, n_unique_times
    )
    risk_set = risk_set_at_time[time_end_return_inverse - 1]

    loss = -np.sum(np.nan_to_num(events * (p - np.log(risk_set))))

    XxXb = np.multiply(X, p_exp[:, np.newaxis])

    XxXb_at_Xt_at_time_cumsum = bincount_reverse_cumsum_along_axis(
        XxXb, time_end_return_inverse, time_start_return_inverse, n_unique_times
    )

    XxXb_at_Xt_at_index = XxXb_at_Xt_at_time_cumsum[time_end_return_inverse - 1]

    XxXb_at_Xt_at_index_div_riskset = XxXb_at_Xt_at_index / risk_set[:, np.newaxis]

    jacobian_presum = events[:, np.newaxis] * (X - XxXb_at_Xt_at_index_div_riskset)

    jacobian_presum = np.nan_to_num(jacobian_presum)

    jacobian = -np.sum(jacobian_presum, axis=0)

    X2Xb_at_Xt_at_time_at_risk = self_outterproduct_mul_groupby_broadcast_to_at_risk(
        X, p_exp, time_end_return_inverse, time_start_return_inverse, n_unique_times
    )

    a = X2Xb_at_Xt_at_time_at_risk / risk_set_at_time[:, None, None]

    b = (
        np.matmul(
            XxXb_at_Xt_at_time_cumsum[:, :, None], XxXb_at_Xt_at_time_cumsum[:, None, :]
        )
        / (risk_set_at_time**2)[:, None, None]
    )

    c = (a - b) * event_counts_at_times[:, None, None]

    c = np.nan_to_num(c)

    hessian = np.sum(c, axis=0)

    return loss, jacobian, hessian



def train_cox_ph_breslow_left_censorship(
    X_strata,
    events_strata,
    n_unique_times_strata,
    event_counts_at_times_strata,
    time_end_return_inverse_strata,
    time_start_return_inverse_strata,
    n_strata,
    alpha,
    l1_ratio,
    weights,
    max_iter,
    tol,
):

    last_loss = np.array(np.inf)

    half_step = False

    for _ in range(max_iter):
        neg_log_likelihoods = []
        neg_log_likelihood_jacobians = []
        neg_log_likelihood_hessians = []
        for s_i in range(n_strata):
            (
                neg_log_likelihood_loss_strata,
                neg_log_likelihood_jacobian_strata,
                neg_log_likelihood_hessian_strata,
            ) = breslow_neg_log_likelihood_loss_jacobian_hessian_left_censorship(
                weights,
                X_strata[s_i],
                events_strata[s_i],
                time_end_return_inverse_strata[s_i],
                time_start_return_inverse_strata[s_i],
                n_unique_times_strata[s_i],
                event_counts_at_times_strata[s_i],
            )
            neg_log_likelihoods.append(neg_log_likelihood_loss_strata)
            neg_log_likelihood_jacobians.append(neg_log_likelihood_jacobian_strata)
            neg_log_likelihood_hessians.append(neg_log_likelihood_hessian_strata)

        neg_log_likelihood_loss = np.sum(neg_log_likelihoods, 0)
        neg_log_likelihood_jacobian = np.sum(neg_log_likelihood_jacobians, 0)
        neg_log_likelihood_hessian = np.sum(neg_log_likelihood_hessians, 0)

        elasticnet_loss, elasticnet_jacobian, elasticnet_hessian = (
            elasticnet_loss_jacobian_hessian(weights, alpha, l1_ratio)
        )

        loss = neg_log_likelihood_loss + elasticnet_loss
        jacobian = neg_log_likelihood_jacobian + elasticnet_jacobian
        hessian = neg_log_likelihood_hessian + elasticnet_hessian

        if abs(last_loss - loss) <= tol:
            break
        elif (loss < last_loss) & (not half_step):
            last_loss = loss
            weights = weights - np.dot(np.linalg.inv(hessian), jacobian)
        elif (loss < last_loss) & half_step:
            last_loss = loss
            weights = weights - (0.5 * np.dot(np.linalg.inv(hessian), jacobian))
        else:
            if half_step:
                break
            else:
                half_step = True

    return weights, loss

