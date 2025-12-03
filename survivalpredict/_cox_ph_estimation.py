import numba as nb
import numpy as np

from ._stratification import split_and_preprocess_data_by_strata


def elasticnet_loss_jacobian_hessian(weights, alpha, l1_ratio):
    l1 = alpha * l1_ratio * np.abs(weights).sum()
    l2 = 0.5 * alpha * (1.0 - l1_ratio) * np.square(weights).sum()
    elasticnet_loss = l1 + l2

    l1_jacobian = np.sign(weights) * alpha * l1_ratio
    l2_jacobian = 2 * weights * (0.5 * alpha) * (1 - l1_ratio)

    elasticnet_jacobian = l1_jacobian + l2_jacobian

    elasticnet_hessian = np.identity(weights.shape[0]) * (alpha - (l1_ratio * alpha))

    return elasticnet_loss, elasticnet_jacobian, elasticnet_hessian


def reverse_cumsum(a):
    return np.flip(np.cumsum(np.flip(a)))


self_outterproduct_mul_groupby_time_sig = nb.types.Array(
    nb.types.float64, 3, "C", False, aligned=True
)(
    nb.types.Array(nb.types.float64, 2, "A", False, aligned=True),
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.int64,
)


@nb.jit(self_outterproduct_mul_groupby_time_sig, cache=True)
def self_outterproduct_mul_groupby_time(X, p_exp, time_return_inverse, max_time_index):

    n_rows = X.shape[0]
    n_col = X.shape[1]

    c = np.empty((n_col, n_col))
    output = np.zeros((max_time_index, n_col, n_col))

    for i in range(n_rows):
        X_i = X[i]
        p_i = p_exp[i]
        t_i = time_return_inverse[i]

        for e in range(n_col):
            for f in range(n_col):
                c[e, f] = X_i[e] * X_i[f] * p_i
        output[t_i] = output[t_i] + c

    return output


def breslow_neg_log_likelihood_loss_jacobian_hessian(
    weights, X, event, time_return_inverse, n_unique_times, event_counts_at_times
):

    p = np.dot(X, weights)
    p_exp = np.exp(p)

    risk_set_at_time = reverse_cumsum(
        np.bincount(time_return_inverse, weights=p_exp, minlength=n_unique_times)
    )

    risk_set = risk_set_at_time[time_return_inverse]

    loss = -np.sum(event * (p - np.log(risk_set)))

    XxXb = np.multiply(X, p_exp[:, np.newaxis])

    XxXb_at_Xt_at_time = np.apply_along_axis(
        lambda a: np.bincount(time_return_inverse, weights=a, minlength=n_unique_times),
        0,
        XxXb,
    )

    del XxXb

    XxXb_at_Xt_at_time_cumsum = np.apply_along_axis(
        reverse_cumsum, 0, XxXb_at_Xt_at_time
    )

    del XxXb_at_Xt_at_time

    XxXb_at_Xt_at_index = XxXb_at_Xt_at_time_cumsum[time_return_inverse]

    jacobian = -np.sum(
        event[:, np.newaxis] * (X - XxXb_at_Xt_at_index / risk_set[:, np.newaxis]),
        axis=0,
    )

    del XxXb_at_Xt_at_index, risk_set

    X2xXb_at_time_ = self_outterproduct_mul_groupby_time(
        X, p_exp, time_return_inverse, n_unique_times
    )

    X2Xb_at_Xt_at_time_cumsum_ = np.flip(np.add.accumulate(np.flip(X2xXb_at_time_)))

    del X2xXb_at_time_

    a = X2Xb_at_Xt_at_time_cumsum_ / risk_set_at_time[:, None, None]

    del X2Xb_at_Xt_at_time_cumsum_

    b = (
        np.matmul(
            XxXb_at_Xt_at_time_cumsum[:, :, None], XxXb_at_Xt_at_time_cumsum[:, None, :]
        )
        / (risk_set_at_time**2)[:, None, None]
    )

    del XxXb_at_Xt_at_time_cumsum, risk_set_at_time

    c = (a - b) * event_counts_at_times[:, None, None]

    del a, b
    hessian = np.sum(c, axis=0)

    return loss, jacobian, hessian


def train_cox_ph_breslow(
    X_strata,
    events_strata,
    n_unique_times_strata,
    event_counts_at_times_strata,
    time_return_inverse_strata,
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
            ) = breslow_neg_log_likelihood_loss_jacobian_hessian(
                weights,
                X_strata[s_i],
                events_strata[s_i],
                time_return_inverse_strata[s_i],
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


get_first_half_of_eforn_hessian_sig = nb.types.Array(
    nb.types.float64, 2, "C", False, aligned=True
)(
    nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.bool_, 1, "C", False, aligned=True),
)


@nb.njit(get_first_half_of_eforn_hessian_sig, cache=True)
def get_first_half_of_efron_hessian(
    jacobian_numerator, risk_set_minus_l_div_m_x_total_risk, events
):
    n_rows = jacobian_numerator.shape[0]
    n_col = jacobian_numerator.shape[1]

    first_half_of_efron_hessian_sum = np.zeros((n_col, n_col))

    rs = np.square(risk_set_minus_l_div_m_x_total_risk)

    for i in range(n_rows):
        jn_i = jacobian_numerator[i]
        rs_i = rs[i]
        e_i = events[i]

        if e_i:
            for e in range(n_col):
                for f in range(n_col):
                    first_half_of_efron_hessian_sum[e, f] += (jn_i[e] * jn_i[f]) / (
                        rs_i
                    )

    return first_half_of_efron_hessian_sum


get_second_half_of_efron_hessian_signature = nb.types.Array(
    nb.float64, 2, "C", False, aligned=True
)(
    nb.types.Array(nb.types.bool_, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.float64, 3, "C", False, aligned=True),
    nb.types.Array(nb.types.float64, 3, "A", False, aligned=True),
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
)


@nb.njit(get_second_half_of_efron_hessian_signature, cache=True)
def get_second_half_of_efron_hessian(
    events,
    time_return_inverse,
    X2xXb_at_time,
    X2Xb_at_Xt_at_time_cumsum,
    l_div_m,
    risk_set_minus_l_div_m_x_total_risk,
):

    n_col = X2xXb_at_time.shape[1]
    n_rows = events.shape[0]

    second_half_of_eforn_hessian_sum = np.zeros((n_col, n_col))

    for i in range(n_rows):
        e_i = events[i]

        if e_i:
            t_i = time_return_inverse[i]
            l_div_m_i = l_div_m[i]
            X2xXb_at_time_i = X2xXb_at_time[t_i]
            X2Xb_at_Xt_at_time_cumsum_i = X2Xb_at_Xt_at_time_cumsum[t_i]
            rsl_div_m_i = risk_set_minus_l_div_m_x_total_risk[i]

            for e in range(n_col):
                for f in range(n_col):
                    second_half_of_eforn_hessian_sum[e, f] += (
                        X2Xb_at_Xt_at_time_cumsum_i[e, f]
                        - l_div_m_i * X2xXb_at_time_i[e, f]
                    ) / rsl_div_m_i

    return second_half_of_eforn_hessian_sum


def efron_neg_log_likelihood_loss_jacobian_hessian(
    weights, X, events, l_div_m, time_return_inverse, n_unique_times
):

    p = np.dot(X, weights)
    p_exp = np.exp(p)

    total_risk_per_at_time = np.bincount(
        time_return_inverse, weights=p_exp, minlength=n_unique_times
    )
    total_risk_per_at_index = total_risk_per_at_time[time_return_inverse]

    risk_set = reverse_cumsum(total_risk_per_at_time)[time_return_inverse]
    risk_set_minus_l_div_m_x_total_risk_per_at_index = risk_set - (
        l_div_m * total_risk_per_at_index
    )

    del risk_set

    loss = -np.sum(
        events * (p - np.log(risk_set_minus_l_div_m_x_total_risk_per_at_index))
    )

    del p, total_risk_per_at_index

    XxP = np.multiply(X, p_exp[:, np.newaxis])
    XxP_per_time = np.apply_along_axis(
        lambda a: np.bincount(time_return_inverse, weights=a, minlength=n_unique_times),
        0,
        XxP,
    )

    del XxP

    XxP_per_time_cumsum = np.apply_along_axis(reverse_cumsum, 0, XxP_per_time)
    XxP_per_time_cumsum_at_index = XxP_per_time_cumsum[time_return_inverse]

    XxP_at_h = XxP_per_time[time_return_inverse]
    l_div_m_times_XxXb_at_Xh = l_div_m[:, None] * XxP_at_h

    del XxP_at_h

    jacobian_numerator = XxP_per_time_cumsum_at_index - l_div_m_times_XxXb_at_Xh

    del XxP_per_time_cumsum_at_index, l_div_m_times_XxXb_at_Xh

    jacobian = -np.sum(
        events[:, np.newaxis]
        * (
            X
            - (
                jacobian_numerator
                / risk_set_minus_l_div_m_x_total_risk_per_at_index[:, None]
            )
        ),
        axis=0,
    )

    first_half_of_efron_hessian = get_first_half_of_efron_hessian(
        jacobian_numerator, risk_set_minus_l_div_m_x_total_risk_per_at_index, events
    )

    X2xXb_at_time = self_outterproduct_mul_groupby_time(
        X, p_exp, time_return_inverse, n_unique_times
    )

    del p_exp

    X2Xb_at_Xt_at_time_cumsum = np.flip(np.add.accumulate(np.flip(X2xXb_at_time)))

    second_half_of_efron_hessian = get_second_half_of_efron_hessian(
        events,
        time_return_inverse,
        X2xXb_at_time,
        X2Xb_at_Xt_at_time_cumsum,
        l_div_m,
        risk_set_minus_l_div_m_x_total_risk_per_at_index,
    )

    del (
        risk_set_minus_l_div_m_x_total_risk_per_at_index,
        X2xXb_at_time,
        X2Xb_at_Xt_at_time_cumsum,
    )

    hessian = -(first_half_of_efron_hessian - second_half_of_efron_hessian)

    return loss, jacobian, hessian


index_per_not_censored_times_nb_signature = nb.types.Array(nb.types.int64, 1, "C")(
    nb.types.Array(nb.types.int64, 1, "C"), nb.types.Array(nb.types.bool_, 1, "C")
)


def train_cox_ph_efron(
    n_strata,
    X_strata,
    events_strata,
    n_unique_times_strata,
    l_div_m_stata,
    time_return_inverse_strata,
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
            ) = efron_neg_log_likelihood_loss_jacobian_hessian(
                weights,
                X_strata[s_i],
                events_strata[s_i],
                l_div_m_stata[s_i],
                time_return_inverse_strata[s_i],
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
