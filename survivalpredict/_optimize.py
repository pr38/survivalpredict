from typing import Any, Callable

import numpy as np
from scipy.linalg import solve


def newton(
    callable_for_loss_jacobian_hessian: Callable,
    args: tuple[Any],
    callable_for_loss: Callable,
    n_iter: int,
    tol: float,
    weights: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> tuple[np.ndarray[tuple[int], np.dtype[np.float64]], float, int]:
    """
    Variation of 'Newton-Raphson' called 'Adaptive Newton'.
    Taken from Konstantin Mishchenko's 'Regularized Newton Method with Global O(1/k2) Convergence'.
    A penalty is added to the solution of the inverse of the hessian, when generating the Newton step.
    Backtracking with half-step is also added as an extra guard. 
    """
    # A note on how this is done elsewhere:
    # scikit-survival uses Newton with scaling the jacobian and hessian by 1/number of observations, and backtracking(if a step increases loss, a half step is taken instead).
    # Statsmodels only scales jacobian and hessian without backtracking.
    # Lifelines uses a form 'Damped Newton', where a dynamic 'learning rate'/'dampener' is added to the step.
    # I have read that R's survival uses Newton with backtracking. I am not able to comprehend R library code, and can't confirm this.
    # Cox-ph is not convex; there is no guarantee that different implementations will converge at the same set of weights.
    # Also, using a straight Newton-Raphson will sometimes not find the minima. Dampening or backtracking can help Newton find the minima.

    n_cols = len(weights)
    identity_matrix = np.identity(n_cols)
    weights_prev = weights
    n_iters_seen = 0
    loss = float("inf")

    # used to add regularization to the solution of the inverse of the hessian
    # this static value is taken from the default value for Regularized Newton at https://github.com/konstmish/opt_methods/tree/master
    # this value may later be exposed as a class parameter
    H = 5e-05

    while True:
        if n_iters_seen >= n_iter:
            break
        
        loss_new, jacobian, hessian = callable_for_loss_jacobian_hessian(weights, *args)

        if n_iters_seen != 0:
            r = np.linalg.norm(weights - weights_prev)
            #convergence criteria from 'Regularized Newton Method' paper.
            if (np.linalg.norm(jacobian) < (2 * penatly * r)) and (
                loss <= (loss_new - (2/3) * penatly * r**2)
            ):
                return weights, loss, n_iters_seen


        n_iters_seen += 1


        penatly = np.sqrt(H * np.linalg.norm(jacobian))
        regularization = identity_matrix * penatly

        delta = solve(hessian + regularization, jacobian, check_finite=False)
        #delta = solve(hessian, jacobian, check_finite=False)


        new_weights = weights - delta

        #next_loss = callable_for_loss(new_weights, *args)

        print(loss_new, loss,loss_new / loss)
        #backtracking with .5 step
        if loss_new > loss: #backtracking with .5 step
            print('half step')
            weights = (weights_prev + weights) / 2
            loss = callable_for_loss(weights, *args)
            continue

        loss_dif = loss - loss_new
        if loss_dif != 0 and loss_dif < tol:
            break
        
        print('full step')
        weights_prev = weights.copy()
        weights = new_weights
        loss = loss_new
        H = H / 2  # increase regularization

    return weights, loss_new, n_iters_seen
