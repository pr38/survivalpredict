import numba as nb
import numpy as np
from scipy.optimize import minimize

from ._cox_ph_estimation import (
    elasticnet_loss_jacobian_hessian,
    get_first_half_of_efron_hessian,
    get_second_half_of_efron_hessian,
    self_outterproduct_mul_groupby_time,
)
from ._optimize import newton

# to do: remove l1 from coxph.


at_risk_per_time_with_start_times_sig = nb.types.Array(
    nb.types.float64, 1, "C", aligned=True
)(
    nb.types.Array(nb.types.float64, 1, "C", aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", aligned=True),
    nb.types.int64,
)


@nb.njit(at_risk_per_time_with_start_times_sig, cache=True)
def at_risk_per_time_with_start_times(
    p_exp, time_end_return_inverse, time_start_return_inverse, n_unique_times
):

    n_rows = len(p_exp)
    output = np.zeros(n_unique_times)

    for i in range(n_rows):
        output[time_start_return_inverse[i] : time_end_return_inverse[i]] += p_exp[i]

    return output


bincount_reverse_cumsum_along_axis_sig = nb.types.Array(
    nb.types.float64, 2, "C", aligned=True
)(
    nb.types.Array(nb.types.float64, 2, "C", aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", aligned=True),
    nb.types.int64,
)


@nb.njit(bincount_reverse_cumsum_along_axis_sig, cache=True)
def bincount_reverse_cumsum_along_axis(
    XxXb, time_end_return_inverse, time_start_return_inverse, n_unique_times
):
    output = np.zeros((n_unique_times, XxXb.shape[1]))
    n_rows = XxXb.shape[0]

    for i in range(n_rows):
        output[time_start_return_inverse[i] : time_end_return_inverse[i]] += XxXb[i]

    return output


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


def breslow_neg_log_likelihood_loss_jacobian_hessian_left_censorship_with_strata_and_penalty(
    weights,
    n_strata,
    X_strata,
    events_strata,
    time_end_return_inverse_strata,
    time_start_return_inverse_strata,
    n_unique_times_strata,
    event_counts_at_times_strata,
    alpha,
    l1_ratio,
):
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

    return loss, jacobian, hessian


def breslow_neg_log_likelihood_loss_left_censorship(
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

    return -np.sum(np.nan_to_num(events * (p - np.log(risk_set))))


def breslow_neg_log_likelihood_loss_left_censorship_with_strata_and_penalty(
    weights,
    n_strata,
    X_strata,
    events_strata,
    time_end_return_inverse_strata,
    time_start_return_inverse_strata,
    n_unique_times_strata,
    event_counts_at_times_strata,
    alpha,
    l1_ratio,
):
    neg_log_likelihoods = []
    for s_i in range(n_strata):
        neg_log_likelihood_loss_strata = (
            breslow_neg_log_likelihood_loss_left_censorship(
                weights,
                X_strata[s_i],
                events_strata[s_i],
                time_end_return_inverse_strata[s_i],
                time_start_return_inverse_strata[s_i],
                n_unique_times_strata[s_i],
                event_counts_at_times_strata[s_i],
            )
        )
        neg_log_likelihoods.append(neg_log_likelihood_loss_strata)

    neg_log_likelihood_loss = np.sum(neg_log_likelihoods, 0)
    elasticnet_loss, _, __ = elasticnet_loss_jacobian_hessian(weights, alpha, l1_ratio)

    loss = neg_log_likelihood_loss + elasticnet_loss

    return loss


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
    args = (
        n_strata,
        X_strata,
        events_strata,
        time_end_return_inverse_strata,
        time_start_return_inverse_strata,
        n_unique_times_strata,
        event_counts_at_times_strata,
        alpha,
        l1_ratio,
    )

    weights, loss, max_iter_seen = newton(
        breslow_neg_log_likelihood_loss_jacobian_hessian_left_censorship_with_strata_and_penalty,
        args,
        breslow_neg_log_likelihood_loss_left_censorship_with_strata_and_penalty,
        max_iter,
        tol,
        weights,
    )

    return weights, loss, max_iter_seen


@nb.njit(
    nb.types.Array(nb.types.float64, 2, "C", False, aligned=True)(
        nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.int64,
    ),
    cache=True,
)
def bincount_along_axis(XxXb, time_end_return_inverse, n_unique_times):
    output = np.zeros((n_unique_times, XxXb.shape[1]))
    n_rows = XxXb.shape[0]

    for i in range(n_rows):
        output[time_end_return_inverse[i]] += XxXb[i]

    return output


@nb.jit(
    nb.types.Tuple(
        (
            nb.types.Array(nb.types.float64, 3, "C", False, aligned=True),
            nb.types.Array(nb.types.float64, 3, "C", False, aligned=True),
        )
    )(
        nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.int64,
    ),
    cache=True,
)
def self_outterproduct_mul_groupby_broadcast_to_at_at_time_risk(
    X, p_exp, time_end_return_inverse, time_start_return_inverse, max_time_index
):

    n_rows = X.shape[0]
    n_col = X.shape[1]

    c = np.empty((n_col, n_col))
    at_risk = np.zeros((max_time_index, n_col, n_col))
    at_time = np.zeros((max_time_index, n_col, n_col))

    for i in range(n_rows):
        X_i = X[i]
        p_i = p_exp[i]

        for e in range(n_col):
            for f in range(n_col):
                c[e, f] = X_i[e] * X_i[f] * p_i
        at_risk[time_start_return_inverse[i] : time_end_return_inverse[i]] += c
        at_time[time_end_return_inverse[i]] += c

    return at_time, at_risk


def safe_log(a):
    return np.log(a, out=np.zeros_like(a, dtype=np.float64), where=(a != 0))


def efron_neg_log_likelihood_loss_jacobian_hessian_left_censorship(
    weights,
    X,
    events,
    time_end_return_inverse,
    time_start_return_inverse,
    l_div_m,
    n_unique_times,
):
    p = np.dot(X, weights)
    p_exp = np.exp(p)

    risk_set_at_time = at_risk_per_time_with_start_times(
        p_exp, time_end_return_inverse, time_start_return_inverse, n_unique_times
    )
    risk_set = risk_set_at_time[time_end_return_inverse - 1]

    total_event_risk_per_at_index = np.bincount(
        time_end_return_inverse - 1, weights=p_exp * events, minlength=n_unique_times
    )[time_end_return_inverse - 1]

    risk_set_minus_l_div_m_x_total_risk_per_at_index = risk_set - (
        l_div_m * total_event_risk_per_at_index
    )

    loss = -np.sum(
        np.nan_to_num(
            events * (p - np.log(risk_set_minus_l_div_m_x_total_risk_per_at_index))
        )
    )

    XxP = np.multiply(X, p_exp[:, np.newaxis])

    # XxP_per_time = bincount_along_axis(XxP, time_end_return_inverse - 1, n_unique_times)
    # XxP_at_h = XxP_per_time[time_end_return_inverse - 1]

    XxP_event_per_time = np.apply_along_axis(
        lambda a: np.bincount(
            time_end_return_inverse - 1, weights=a, minlength=n_unique_times
        ),
        0,
        XxP * events[:, np.newaxis],
    )

    XxP_per_time_cumsum = bincount_reverse_cumsum_along_axis(
        XxP, time_end_return_inverse, time_start_return_inverse, n_unique_times
    )
    XxP_per_time_cumsum_at_index = XxP_per_time_cumsum[time_end_return_inverse - 1]

    l_div_m_times_XxXb_at_Xh = (
        l_div_m[:, None] * XxP_event_per_time[time_end_return_inverse - 1]
    )

    jacobian_numerator = XxP_per_time_cumsum_at_index - l_div_m_times_XxXb_at_Xh

    jacobian_presum = events[:, np.newaxis] * (
        X
        - (
            jacobian_numerator
            / risk_set_minus_l_div_m_x_total_risk_per_at_index[:, None]
        )
    )

    jacobian_presum = np.nan_to_num(jacobian_presum)

    jacobian = -np.sum(jacobian_presum, axis=0)

    first_half_of_efron_hessian = get_first_half_of_efron_hessian(
        jacobian_numerator, risk_set_minus_l_div_m_x_total_risk_per_at_index, events
    )

    _, X2xXb_at_risk = self_outterproduct_mul_groupby_broadcast_to_at_at_time_risk(
        X, p_exp, time_end_return_inverse, time_start_return_inverse, n_unique_times
    )

    X2xXb_events_at_time = self_outterproduct_mul_groupby_time(
        X, p_exp * events, time_end_return_inverse, n_unique_times
    )

    second_half_of_efron_hessian = get_second_half_of_efron_hessian(
        events,
        time_end_return_inverse - 1,
        X2xXb_events_at_time,
        X2xXb_at_risk,
        l_div_m,
        risk_set_minus_l_div_m_x_total_risk_per_at_index,
    )

    hessian = -(first_half_of_efron_hessian - second_half_of_efron_hessian)

    return loss, jacobian, hessian


def efron_neg_log_likelihood_loss_jacobian_hessian_left_censorship_with_strata_and_penalty(
    weights,
    n_strata,
    X_strata,
    events_strata,
    n_unique_times_strata,
    l_div_m_stata,
    time_end_return_inverse_strata,
    time_start_return_inverse_strata,
    alpha,
    l1_ratio,
):

    neg_log_likelihoods = []
    neg_log_likelihood_jacobians = []
    neg_log_likelihood_hessians = []
    for s_i in range(n_strata):
        (
            neg_log_likelihood_loss_strata,
            neg_log_likelihood_jacobian_strata,
            neg_log_likelihood_hessian_strata,
        ) = efron_neg_log_likelihood_loss_jacobian_hessian_left_censorship(
            weights,
            X_strata[s_i],
            events_strata[s_i],
            time_end_return_inverse_strata[s_i],
            time_start_return_inverse_strata[s_i],
            l_div_m_stata[s_i],
            n_unique_times_strata[s_i],
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

    return loss, jacobian, hessian


def efron_neg_log_likelihood_loss(
    weights,
    X,
    events,
    time_end_return_inverse,
    time_start_return_inverse,
    l_div_m,
    n_unique_times,
):
    p = np.dot(X, weights)
    p_exp = np.exp(p)

    risk_set_at_time = at_risk_per_time_with_start_times(
        p_exp, time_end_return_inverse, time_start_return_inverse, n_unique_times
    )
    risk_set = risk_set_at_time[time_end_return_inverse - 1]

    total_event_risk_per_at_index = np.bincount(
        time_end_return_inverse - 1, weights=p_exp * events, minlength=n_unique_times
    )[time_end_return_inverse - 1]

    risk_set_minus_l_div_m_x_total_risk_per_at_index = risk_set - (
        l_div_m * total_event_risk_per_at_index
    )

    return -np.sum(
        events * (p - safe_log(risk_set_minus_l_div_m_x_total_risk_per_at_index))
    )


def efron_neg_log_likelihood_loss_left_censorship_with_strata_and_penalty(
    weights,
    n_strata,
    X_strata,
    events_strata,
    n_unique_times_strata,
    l_div_m_stata,
    time_end_return_inverse_strata,
    time_start_return_inverse_strata,
    alpha,
    l1_ratio,
):
    neg_log_likelihoods = []

    for s_i in range(n_strata):
        neg_log_likelihood_loss_strata = efron_neg_log_likelihood_loss(
            weights,
            X_strata[s_i],
            events_strata[s_i],
            time_end_return_inverse_strata[s_i],
            time_start_return_inverse_strata[s_i],
            l_div_m_stata[s_i],
            n_unique_times_strata[s_i],
        )
        neg_log_likelihoods.append(neg_log_likelihood_loss_strata)

    neg_log_likelihood_loss = np.sum(neg_log_likelihoods, 0)

    elasticnet_loss, elasticnet_jacobian, elasticnet_hessian = (
        elasticnet_loss_jacobian_hessian(weights, alpha, l1_ratio)
    )

    return neg_log_likelihood_loss + elasticnet_loss


def train_cox_ph_efron_left_censorship(
    n_strata,
    X_strata,
    events_strata,
    n_unique_times_strata,
    l_div_m_stata,
    time_end_return_inverse_strata,
    time_start_return_inverse_strata,
    alpha,
    l1_ratio,
    weights,
    max_iter,
    tol,
):
    args = (
        n_strata,
        X_strata,
        events_strata,
        n_unique_times_strata,
        l_div_m_stata,
        time_end_return_inverse_strata,
        time_start_return_inverse_strata,
        alpha,
        l1_ratio,
    )

    weights, loss, max_iter_seen = newton(
        efron_neg_log_likelihood_loss_jacobian_hessian_left_censorship_with_strata_and_penalty,
        args,
        efron_neg_log_likelihood_loss_left_censorship_with_strata_and_penalty,
        max_iter,
        tol,
        weights,
    )

    return weights, loss, max_iter_seen
