import numba as nb
import numpy as np

from ._stratification import _unique_with_return_inverse

get_breslow_neg_log_likelihood_with_elasticnet_penalty_signature = nb.types.float64(
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
    nb.types.Array(nb.types.bool_, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.int64,
    nb.types.float64,
    nb.types.float64,
    nb.types.bool_,
)


@nb.njit(get_breslow_neg_log_likelihood_with_elasticnet_penalty_signature, cache=True)
def get_breslow_neg_log_likelihood_with_elasticnet_penalty(
    weights,
    X,
    event,
    time_return_inverse,
    n_unique_times,
    alpha,
    l1_ratio,
    scaled,
):
    n = float(X.shape[0])

    p = np.dot(X, weights)
    p_exp = np.exp(p)

    risk_set_at_time = np.flip(
        np.cumsum(
            np.flip(
                np.bincount(
                    time_return_inverse, weights=p_exp, minlength=n_unique_times
                )
            )
        )
    )

    risk_set = risk_set_at_time[time_return_inverse]

    breslow_neg_log_likelihood = -np.sum(event * (p - np.log(risk_set)))

    l1 = alpha * l1_ratio * np.abs(weights).sum()
    l2 = 0.5 * alpha * (1.0 - l1_ratio) * np.square(weights).sum()
    elasticnet_loss = l1 + l2

    if scaled:
        return (2 / n * breslow_neg_log_likelihood) + elasticnet_loss
    else:
        return breslow_neg_log_likelihood + elasticnet_loss


@nb.njit
def soft_threasholding_operator(z, t):
    return np.fmax((np.abs(z) - t), 0) * np.sign(z)


train_cox_elastic_net_signature = nb.types.Tuple(
    (
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.float64,
    )
)(
    nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.bool_, 1, "C", False, aligned=True),
    nb.types.float64,
    nb.types.float64,
    nb.types.float64,
    nb.types.int64,
)


@nb.njit(train_cox_elastic_net_signature, cache=True)
def train_cox_elastic_net_regularization_paths(
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    alpha: float,
    l1_ratio: float,
    tol: float,
    n_iter: int,
) -> tuple[np.ndarray, float]:
    """A direct implementation of,
    Simon N, Friedman J, Hastie T, Tibshirani R. Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent. J Stat Softw. 2011 Mar;39(5):1-13. doi: 10.18637/jss.v039.i05. PMID: 27065756; PMCID: PMC4824408.
    """
    unique_times, time_return_inverse = _unique_with_return_inverse(times)
    n_unique_times = len(unique_times)

    n = X.shape[0]

    col_indexes = np.arange(X.shape[1])

    weights = np.zeros(X.shape[1])

    last_loss = np.inf

    for _ in range(n_iter):

        new_weights = weights.copy()
        n_hat = np.dot(X, new_weights)
        n_hat_exp = np.exp(n_hat)
        n_hat_exp_risk_set = np.flip(
            np.cumsum(
                np.flip(
                    np.bincount(
                        time_return_inverse, n_hat_exp, minlength=n_unique_times
                    )
                )
            )
        )[time_return_inverse]
        wn = np.sum(
            events
            * (
                (n_hat_exp * n_hat_exp_risk_set - n_hat_exp**2)
                / (n_hat_exp_risk_set**2)
            )
        )
        zn = n_hat + 1 / wn * (
            events - np.sum(events * (n_hat_exp / n_hat_exp_risk_set))
        )
        for j in range(X.shape[1]):
            not_j = col_indexes[col_indexes != j]
            not_j_n = np.dot(X[:, not_j], new_weights[not_j])
            left = 1 / n * np.sum(wn * X[:, j] * (zn - not_j_n))
            right = alpha * l1_ratio
            top = soft_threasholding_operator(left, right)
            bottom = 1 / n * np.sum(wn * (X[:, j]) ** 2 + alpha * (1 - l1_ratio))
            new_weights[j] = top / bottom

        loss = get_breslow_neg_log_likelihood_with_elasticnet_penalty(
            weights,
            X,
            events,
            time_return_inverse,
            n_unique_times,
            alpha,
            l1_ratio,
            True,
        )

        if (loss - last_loss) > tol:
            break
        last_loss = loss
        weights = new_weights

    final_loss = get_breslow_neg_log_likelihood_with_elasticnet_penalty(
        weights, X, events, time_return_inverse, n_unique_times, alpha, l1_ratio, False
    )

    return weights, final_loss


get_breslow_neg_log_likelihood_with_elasticnet_penalty_with_left_censorship_signature = nb.types.float64(
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
    nb.types.Array(nb.types.bool_, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.int64,
    nb.types.float64,
    nb.types.float64,
    nb.types.bool_,
)


@nb.njit(
    get_breslow_neg_log_likelihood_with_elasticnet_penalty_with_left_censorship_signature,
    cache=True,
)
def get_breslow_neg_log_likelihood_with_elasticnet_penalty_with_left_censorship(
    weights,
    X,
    event,
    time_end_return_inverse,
    time_start_return_inverse,
    n_unique_times,
    alpha,
    l1_ratio,
    scaled,
):
    n = float(X.shape[0])

    p = np.dot(X, weights)
    p_exp = np.exp(p)

    risk_removed_at_time = np.bincount(
        time_end_return_inverse, weights=p_exp, minlength=n_unique_times
    )
    risk_added_at_time = np.bincount(
        time_start_return_inverse, weights=p_exp, minlength=n_unique_times
    )
    risk_at_time = np.cumsum(risk_added_at_time - risk_removed_at_time)
    risk_set = risk_at_time[time_end_return_inverse - 1]

    breslow_neg_log_likelihood = -np.sum(event * (p - np.log(risk_set)))

    l1 = alpha * l1_ratio * np.abs(weights).sum()
    l2 = 0.5 * alpha * (1.0 - l1_ratio) * np.square(weights).sum()
    elasticnet_loss = l1 + l2

    if scaled:
        return (2 / n * breslow_neg_log_likelihood) + elasticnet_loss
    else:
        return breslow_neg_log_likelihood + elasticnet_loss


train_cox_elastic_net_with_left_censorship_signature = nb.types.Tuple(
    (
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.float64,
    )
)(
    nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.bool_, 1, "C", False, aligned=True),
    nb.types.float64,
    nb.types.float64,
    nb.types.float64,
    nb.types.int64,
)


@nb.njit(train_cox_elastic_net_with_left_censorship_signature, cache=True)
def train_cox_elastic_net_with_left_censorship(
    X: np.ndarray,
    times: np.ndarray,
    times_start: np.ndarray,
    events: np.ndarray,
    alpha: float,
    l1_ratio: float,
    tol: float,
    n_iter: int,
) -> tuple[np.ndarray, float]:
    """A direct implementation of,
    Simon N, Friedman J, Hastie T, Tibshirani R. Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent. J Stat Softw. 2011 Mar;39(5):1-13. doi: 10.18637/jss.v039.i05. PMID: 27065756; PMCID: PMC4824408.
    """
    all_times = np.concatenate((times_start, times))
    unique_times, unique_times_return_inverse = _unique_with_return_inverse(all_times)

    time_end_return_inverse = unique_times_return_inverse[len(times) :]
    time_start_return_inverse = unique_times_return_inverse[: len(times_start)]

    n_unique_times = len(unique_times)

    n = X.shape[0]

    col_indexes = np.arange(X.shape[1])

    weights = np.zeros(X.shape[1])

    last_loss = np.inf

    for _ in range(n_iter):

        new_weights = weights.copy()
        n_hat = np.dot(X, new_weights)

        n_hat_exp = np.exp(n_hat)

        # n_hat_exp_risk_set = np.flip(
        #     np.cumsum(
        #         np.flip(
        #             np.bincount(
        #                 time_return_inverse, n_hat_exp, minlength=n_unique_times
        #             )
        #         )
        #     )
        # )[time_return_inverse]

        risk_removed_at_time = np.bincount(
            time_end_return_inverse, weights=n_hat_exp, minlength=n_unique_times
        )
        risk_added_at_time = np.bincount(
            time_start_return_inverse, weights=n_hat_exp, minlength=n_unique_times
        )
        risk_at_time = np.cumsum(risk_added_at_time - risk_removed_at_time)
        n_hat_exp_risk_set = risk_at_time[time_end_return_inverse - 1]

        wn = np.sum(
            events
            * (
                (n_hat_exp * n_hat_exp_risk_set - n_hat_exp**2)
                / (n_hat_exp_risk_set**2)
            )
        )
        zn = n_hat + 1 / wn * (
            events - np.sum(events * (n_hat_exp / n_hat_exp_risk_set))
        )
        for j in range(X.shape[1]):
            not_j = col_indexes[col_indexes != j]
            not_j_n = np.dot(X[:, not_j], new_weights[not_j])
            left = 1 / n * np.sum(wn * X[:, j] * (zn - not_j_n))
            right = alpha * l1_ratio
            top = soft_threasholding_operator(left, right)
            bottom = 1 / n * np.sum(wn * (X[:, j]) ** 2 + alpha * (1 - l1_ratio))
            new_weights[j] = top / bottom

        loss = (
            get_breslow_neg_log_likelihood_with_elasticnet_penalty_with_left_censorship(
                weights,
                X,
                events,
                time_end_return_inverse,
                time_start_return_inverse,
                n_unique_times,
                alpha,
                l1_ratio,
                True,
            )
        )

        if (loss - last_loss) > tol:
            break
        last_loss = loss
        weights = new_weights

    final_loss = (
        get_breslow_neg_log_likelihood_with_elasticnet_penalty_with_left_censorship(
            weights,
            X,
            events,
            time_end_return_inverse,
            time_start_return_inverse,
            n_unique_times,
            alpha,
            l1_ratio,
            False,
        )
    )

    return weights, final_loss
