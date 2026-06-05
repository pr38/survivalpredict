from typing import Any, Callable

import numpy as np


def newton_ralphson_with_half_step(
    callable_for_loss_jacobian_hessian: Callable,
    args: tuple[Any],
    callable_for_loss: Callable,
    n_iter: int,
    tol: float,
    weights: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> tuple[np.ndarray[tuple[int], np.dtype[np.float64]], float, int]:
    weights_prev = weights.copy()
    last_loss = np.inf

    n_iters_seen = 0

    while True:

        if n_iters_seen >= n_iter:
            break

        n_iters_seen += 1

        (
            loss,
            jacobian,
            hessian,
        ) = callable_for_loss_jacobian_hessian(weights, *args)

        delta = np.dot(np.linalg.inv(hessian), jacobian)  # newton step

        if loss > last_loss:
            weights_half_step = (weights_prev + weights) / 2
            half_step_loss = callable_for_loss(weights_half_step, *args)

            if half_step_loss < loss:
                weights_prev = weights
                weights = weights_half_step
                loss = half_step_loss
                last_loss = loss
                n_iters_seen += 1

                continue

            else:
                break

        loss_dif = abs(last_loss - loss)

        # to do, add early ending criteria

        if (loss_dif != 0.0) & (
            loss_dif <= tol
        ):  # If loss_dif is exactly 0, keep going. This is simply a quirk feature of training cox with newton.
            return weights_prev, last_loss, n_iters_seen

        weights_prev = weights
        weights = weights - delta
        last_loss = loss

    return weights, loss, n_iters_seen
