import numba as nb
import numpy as np
import pytensor
import pytensor.tensor as pt


def _reverse_cumsum_pt(a):
    return pt.flip(pt.cumsum(pt.flip(a)))


def get_breslow_neg_log_likelihood_loss_jacobian_hessian_function_pytensor() -> pytensor.compile.function.types.Function:

    weights = pt.vector("weights", dtype="float64")
    data = pt.matrix("data", dtype="float64")
    events = pt.vector("event", dtype="float64")
    n_unique_times = pt.scalar("n_unique_times", dtype="int64")
    time_return_inverse = pt.vector("time_return_inverse", dtype="int64")

    alpha = pt.scalar("alpha", dtype="float64")
    l1_ratio = pt.scalar("l1_ratio", dtype="float64")

    l1 = alpha * l1_ratio * pt.abs(weights).sum()
    l2 = 0.5 * alpha * (1.0 - l1_ratio) * pt.square(weights).sum()

    o = pt.dot(data, weights)
    risk_set = _reverse_cumsum_pt(
        pt.bincount(time_return_inverse, weights=pt.exp(o), minlength=n_unique_times)
    )[time_return_inverse]
    loss = -pt.sum(events * (o - pt.log(risk_set))) + l1 + l2

    jacobian = pytensor.gradient.jacobian(loss, weights)
    hessian = pytensor.gradient.hessian(loss, weights)
    neg_log_likelihood_loss_jacobian_hessian = pytensor.function(
        inputs=[
            weights,
            data,
            events,
            alpha,
            l1_ratio,
            n_unique_times,
            time_return_inverse,
        ],
        outputs=[loss, jacobian, hessian],
        mode="NUMBA",
    )

    return neg_log_likelihood_loss_jacobian_hessian


breslow_neg_log_likelihood_loss_jacobian_hessian = (
    get_breslow_neg_log_likelihood_loss_jacobian_hessian_function_pytensor()
)


def train_cox_ph_breslow(X, times, events, alpha, l1_ratio, weights, max_iter, tol):
    unique_times, time_return_inverse = np.unique(times, return_inverse=True)
    n_unique_times = len(unique_times)

    last_loss = np.array(np.inf)

    half_step = False

    for _ in range(max_iter):
        loss, jacobian, hessian = breslow_neg_log_likelihood_loss_jacobian_hessian(
            weights, X, events, alpha, l1_ratio, n_unique_times, time_return_inverse
        )

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


def get_efron_neg_log_likelihood_loss_jacobian_hessian_function() -> pytensor.compile.function.types.Function:

    weights = pt.vector("weights", dtype="float64")
    data = pt.matrix("data", dtype="float64")
    l_div_m = pt.vector("l_div_m", dtype="float64")
    n_unique_times = pt.scalar("n_unique_times", dtype="int64")
    events = pt.vector("event", dtype="int64")
    time_return_inverse = pt.vector("time_return_inverse", dtype="int64")

    alpha = pt.scalar("alpha", dtype="float64")
    l1_ratio = pt.scalar("l1_ratio", dtype="float64")

    l1 = alpha * l1_ratio * pt.abs(weights).sum()
    l2 = 0.5 * alpha * (1.0 - l1_ratio) * pt.square(weights).sum()

    p = pt.dot(data, weights)
    p_exp = pt.exp(p)

    set_at_time_indexed_at_time = pt.bincount(
        time_return_inverse, weights=p_exp, minlength=n_unique_times
    )
    set_per_time = set_at_time_indexed_at_time[time_return_inverse]
    risk_set = _reverse_cumsum_pt(set_at_time_indexed_at_time)[time_return_inverse]

    loss = -pt.sum(events * (p - np.log(risk_set - (l_div_m * set_per_time)))) + l1 + l2

    jacobian = pytensor.gradient.jacobian(loss, weights)
    hessian = pytensor.gradient.hessian(loss, weights)
    neg_log_likelihood_loss_jacobian_hessian = pytensor.function(
        inputs=[
            weights,
            data,
            events,
            alpha,
            l1_ratio,
            l_div_m,
            n_unique_times,
            time_return_inverse,
        ],
        outputs=[loss, jacobian, hessian],
        mode="NUMBA",
    )

    return neg_log_likelihood_loss_jacobian_hessian


efron_neg_log_likelihood_loss_jacobian_hessian = (
    get_efron_neg_log_likelihood_loss_jacobian_hessian_function()
)


index_per_not_censored_times_nb_signature = nb.types.Array(nb.types.int64, 1, "C")(
    nb.types.Array(nb.types.int64, 1, "C"), nb.types.Array(nb.types.bool, 1, "C")
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
        loss, jacobian, hessian = efron_neg_log_likelihood_loss_jacobian_hessian(
            weights,
            X,
            events,
            alpha,
            l1_ratio,
            l_div_m,
            n_unique_times,
            time_return_inverse,
        )

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
