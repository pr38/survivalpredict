from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


def get_mtlr_loss(weights, X, survived_at_times, not_censored_at_times, c1, c2):
    product = jnp.dot(X, weights.T)

    survival_estimation = ((1 + jnp.exp(product)) ** -1).cumprod(axis=1)

    brown_loss = (
        (
            (survived_at_times * jnp.square(1 - survival_estimation))
            + ((1 - survived_at_times) * jnp.square(survival_estimation))
        )
        * not_censored_at_times
    ).sum()

    l2_regularization = c2 / 2 * jnp.sum(jnp.abs(weights))

    diff_between_times_smothing_regularization = (
        c1 / 2 * jnp.sum(jnp.abs(jnp.diff(weights, axis=1)))
    )

    return brown_loss + l2_regularization + diff_between_times_smothing_regularization


def train_mtlr(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
    c1=0.0,
    c2=0.0,
    solver="L-BFGS-B",
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]], float, int]:
    times_max = int(times.max())
    n_cols = X.shape[1]

    times_of_intrest = np.arange(1, times_max + 1)

    survived_at_times = times[:, np.newaxis] >= times_of_intrest
    not_censored_at_times = np.logical_or(
        survived_at_times, events.astype(np.bool_)[:, np.newaxis]
    )

    if times_start is not None:
        not_left_censored = ~times_of_intrest <= times_start[:, np.newaxis]
        not_censored_at_times = np.logical_and(not_censored_at_times, not_left_censored)

    weights = np.zeros((times_max, n_cols))

    # generating the grad function in main scope will trigger unnecessary compute on import
    get_mtlr_grad = jax.grad(get_mtlr_loss)

    def get_mtlr_loss_flattened_weights(
        weights_flatten, X, survived_at_times, not_censored_at_times, c1, c2, times_max
    ):
        n_cols = X.shape[1]
        weights = weights_flatten.reshape(times_max, n_cols)
        return get_mtlr_loss(
            weights, X, survived_at_times, not_censored_at_times, c1, c2
        )

    def get_mtlr_grad_flattened_weights(
        weights_flatten, X, survived_at_times, not_censored_at_times, c1, c2, times_max
    ):
        n_cols = X.shape[1]
        weights = weights_flatten.reshape(times_max, n_cols)
        return get_mtlr_grad(
            weights, X, survived_at_times, not_censored_at_times, c1, c2
        ).flatten()

    args = (X, survived_at_times, not_censored_at_times, c1, c2, times_max)

    result = minimize(
        get_mtlr_loss_flattened_weights,
        weights.flatten(),
        args=args,
        jac=get_mtlr_grad_flattened_weights,
        method=solver,
    )

    weights = result.x.reshape(times_max, n_cols)
    final_loss = result.fun
    max_iter_seen = result.nit

    return weights, final_loss, max_iter_seen


def predict_mtlr(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    weights: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    product = np.dot(X, weights.T)

    return ((1 + np.exp(product)) ** -1).cumprod(axis=1)


def predict_hazard_mtlr(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    weights: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    product = np.dot(X, weights.T)

    return 1 - (1 + np.exp(product)) ** -1


def convert_hazard_to_survival_mtlr(
    hazards: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:

    n_dims = len(hazards.shape)

    if n_dims == 2:
        return (-1 * (hazards - 1)).cumprod(axis=1)
    elif n_dims == 1:
        return (-1 * (hazards - 1)).cumprod()
    else:
        raise ValueError("hazards array should be 1 or 2 dimensional")
