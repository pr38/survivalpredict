import numba as nb
import numpy as np


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


@nb.njit(
    nb.types.Array(nb.float64, 3, "A", False, aligned=True)(
        nb.types.Array(nb.float64, 3, "A", False, aligned=True),
        nb.types.Array(nb.int64, 1, "C", False, aligned=True),
        nb.int64,
    )
)
def three_dimensional_groupby_sum(array, by, n_unique_times):
    output = np.zeros((n_unique_times, array.shape[1], array.shape[2]))

    for i in range(by.shape[0]):
        by_i = by[i]
        array_i = array[i]
        output[by_i] += array_i

    return output


self_outterproduct_mul_scalar_group_by_time_sig = nb.types.Array(
    nb.types.float64, 3, "C", False, aligned=True
)(
    nb.types.Array(nb.types.float64, 2, "F", False, aligned=True),
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.int64,
)


@nb.jit(self_outterproduct_mul_scalar_group_by_time_sig)
def self_outterproduct_mul_scalar_group_by_time(
    X, p_exp, time_return_inverse, max_time_index
):

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

    X2xXb_at_time_ = self_outterproduct_mul_scalar_group_by_time(
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


    del XxXb_at_Xt_at_time_cumsum,risk_set_at_time

    c = (a - b) * event_counts_at_times[:, None, None]

    del a, b
    hessian = np.sum(c, axis=0)

    return loss, jacobian, hessian


def train_cox_ph_breslow(X, times, events, alpha, l1_ratio, weights, max_iter, tol):
    unique_times, time_return_inverse = np.unique(times, return_inverse=True)
    n_unique_times = len(unique_times)
    event_counts_at_times = np.bincount(
        time_return_inverse, weights=events.astype(np.int64)
    )

    last_loss = np.array(np.inf)

    half_step = False

    for _ in range(max_iter):
        (
            neg_log_likelihood_loss,
            neg_log_likelihood_jacobian,
            neg_log_likelihood_hessian,
        ) = breslow_neg_log_likelihood_loss_jacobian_hessian(
            weights,
            X,
            events,
            time_return_inverse,
            n_unique_times,
            event_counts_at_times,
        )
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

    loss = -np.sum(
        events * (p - np.log(risk_set_minus_l_div_m_x_total_risk_per_at_index))
    )

    XxP = np.multiply(X, p_exp[:, np.newaxis])
    XxP_per_time = np.apply_along_axis(
        lambda a: np.bincount(time_return_inverse, weights=a, minlength=n_unique_times),
        0,
        XxP,
    )
    XxP_per_time_cumsum = np.apply_along_axis(reverse_cumsum, 0, XxP_per_time)
    XxP_per_time_cumsum_at_index = XxP_per_time_cumsum[time_return_inverse]

    XxP_at_h = XxP_per_time[time_return_inverse]
    l_div_m_times_XxXb_at_Xh = l_div_m[:, None] * XxP_at_h

    jacobian_numerator = XxP_per_time_cumsum_at_index - l_div_m_times_XxXb_at_Xh

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

    zjlm_out_zjlm = np.matmul(
        jacobian_numerator[:, :, np.newaxis], jacobian_numerator[:, np.newaxis, :]
    )
    a = zjlm_out_zjlm / (
        np.square(risk_set_minus_l_div_m_x_total_risk_per_at_index)[:, None, None]
    )

    XXxP = np.einsum("ij,ik,i->ijk", X, X, p_exp)
    XXxP_at_time = three_dimensional_groupby_sum(
        XXxP, time_return_inverse, n_unique_times
    )
    XXxP_at_time_cumsum_at_index = np.flip(np.add.accumulate(np.flip(XXxP_at_time)))[
        time_return_inverse
    ]

    XXxP_h = XXxP_at_time[time_return_inverse]

    b_numerator = XXxP_at_time_cumsum_at_index - l_div_m[:, None, None] * XXxP_h
    b = b_numerator / (risk_set_minus_l_div_m_x_total_risk_per_at_index[:, None, None])

    hessian = np.sum(events[:, None, None] * (b - a), axis=0)

    return loss, jacobian, hessian


index_per_not_censored_times_nb_signature = nb.types.Array(nb.types.int64, 1, "C")(
    nb.types.Array(nb.types.int64, 1, "C"), nb.types.Array(nb.types.bool_, 1, "C")
)


@nb.njit(index_per_not_censored_times_nb_signature)
def get_index_per_not_censored_times(time, events):
    last_time = np.inf
    current_index = 0
    indexes = []

    for i in range(time.shape[0]):
        t = time[i]
        e = events[i]

        not_censored = e == 1

        if t == last_time and not_censored:
            current_index = current_index + 1

        elif not_censored:
            current_index = 0
        else:
            current_index = -1

        last_time = t

        indexes.append(current_index)

    return np.array(indexes)


def train_cox_ph_efron(X, times, events, alpha, l1_ratio, weights, max_iter, tol):

    argsort = times.argsort(kind="mergesort")

    times = times[argsort]
    events = events[argsort]
    X = X[argsort]

    unique_times, time_return_inverse = np.unique(times, return_inverse=True)
    n_unique_times = len(unique_times)

    death_per_time = np.bincount(
        time_return_inverse, weights=events, minlength=n_unique_times
    )[time_return_inverse]

    index_per_not_censored_times = get_index_per_not_censored_times(times, events)
    l_div_m = np.divide(
        np.array(index_per_not_censored_times),
        death_per_time,
        out=np.zeros(X.shape[0]),
        where=np.logical_and(death_per_time != 0, events == 1),
    )

    last_loss = np.array(np.inf)

    half_step = False

    for _ in range(max_iter):
        (
            neg_log_likelihood_loss,
            neg_log_likelihood_jacobian,
            neg_log_likelihood_hessian,
        ) = efron_neg_log_likelihood_loss_jacobian_hessian(
            weights,
            X,
            events,
            l_div_m,
            time_return_inverse,
            n_unique_times,
        )
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
