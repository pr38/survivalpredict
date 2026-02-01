from functools import reduce
from itertools import pairwise
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax import GradientTransformationExtraArgs


def get_gradient_updater(
    gradient_updater: Literal[
        "adadelta",
        "adagrad",
        "adam",
        "adamax",
        "rmsprop",
    ] = "adam",
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 0.0000001,
    rho: float = 0.95,
    decay: float = 0.9,
) -> GradientTransformationExtraArgs:
    if gradient_updater == "adadelta":
        return optax.adadelta(learning_rate=learning_rate, rho=rho, esp=epsilon)

    elif gradient_updater == "adagrad":
        return optax.adagrad(learning_rate=learning_rate, eps=epsilon)

    elif gradient_updater == "adam":
        return optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2, eps=epsilon)

    elif gradient_updater == "adamax":
        return optax.adamax(
            learning_rate=learning_rate, b1=beta1, b2=beta1, eps=epsilon
        )

    else:  # gradient_updater == 'rmsprop':
        return optax.rmsprop(learning_rate=learning_rate, decay=decay, eps=epsilon)


def relu_jax(x):
    return jnp.where(x < 0, 0, x)


def _reverse_cumsum_pt(a):
    return jnp.flip(jnp.cumsum(jnp.flip(a)))


def _get_cox_net_ph_loss(
    weights, X, events, time_return_inverse, n_unique_times, alpha=0.0, l1_ratio=0.5
):
    matrixs_for_reduce_dot = [X] + weights[:-1]
    second_to_last_layer = reduce(
        lambda a, b: relu_jax(
            jnp.dot(
                a,
                b,
            )
        ),
        matrixs_for_reduce_dot,
    )

    o = jnp.dot(second_to_last_layer, weights[-1])

    risk_set = _reverse_cumsum_pt(
        jnp.bincount(time_return_inverse, weights=jnp.exp(o), minlength=n_unique_times)
    )[time_return_inverse]

    abs_weights_sum = reduce(lambda a, b: a + b, [jnp.sum(jnp.abs(w)) for w in weights])
    square_weights_sum = reduce(
        lambda a, b: a + b, [jnp.sum(jnp.square(w)) for w in weights]
    )

    l1 = alpha * l1_ratio * abs_weights_sum
    l2 = 0.5 * alpha * (1.0 - l1_ratio) * square_weights_sum

    loss = -jnp.sum(events * (o - jnp.log(risk_set)))
    return loss + l1 + l2


def train_cox_net_ph(
    X,
    times,
    events,
    hidden_layers: list[int],
    weights: Optional[list[np.ndarray]] = None,
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
    init_dis: Literal["uniform", "normal"] = "uniform",
    track_loss=True,
    max_iter=100,
    gradient_updater: Literal[
        "adadelta",
        "adagrad",
        "adam",
        "adamax",
        "rmsprop",
    ] = "adam",
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 0.0000001,
    rho: float = 0.95,
    decay: float = 0.9,
) -> tuple[list[np.ndarray], float, list[float]]:

    if weights is None:
        # if not reusing weights, create inital weights
        input_n_cols = X.shape[1]
        weight_matrix_shapes = list(pairwise([input_n_cols] + hidden_layers))
        if init_dis == "uniform":
            initializer = jax.nn.initializers.he_uniform()
        else:
            initializer = jax.nn.initializers.he_normal()

        jax_key = jax.random.key(np.random.randint(-10000, 10000))

        weight_matrix_shapes.append((hidden_layers[-1], 1))
        weights = [initializer(jax_key, shape=ws) for ws in weight_matrix_shapes]
        weights[-1] = weights[-1].flatten()

    unique_times, time_return_inverse = np.unique(times, return_inverse=True)
    n_unique_times = len(unique_times)

    grad_updater = get_gradient_updater(
        gradient_updater, learning_rate, beta1, beta2, epsilon, rho, decay
    )

    opt_state = grad_updater.init(weights)
    get_cox_net_ph_grad = jax.grad(_get_cox_net_ph_loss)

    losses_per_steps = []
    loss = None

    for i in range(max_iter):
        jacobian = get_cox_net_ph_grad(
            weights, X, events, time_return_inverse, n_unique_times, alpha, l1_ratio
        )
        updates, opt_state = grad_updater.update(jacobian, opt_state, weights)
        weights = optax.apply_updates(weights, updates)

        if track_loss:
            loss = _get_cox_net_ph_loss(
                weights, X, events, time_return_inverse, n_unique_times, alpha, l1_ratio
            ).item()
            losses_per_steps.append(loss)

    if loss is None:
        loss = _get_cox_net_ph_loss(
            weights, X, events, time_return_inverse, n_unique_times, alpha, l1_ratio
        ).item()

    weights_np = [np.array(w) for w in weights]

    return weights_np, loss, losses_per_steps


def relu_np(x):
    return np.where(x < 0, 0, x)


def get_relative_risk_from_cox_net_ph_weights(
    X: np.ndarray[tuple[int, int], np.dtype[np.floating]], weights: list[np.ndarray]
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    matrixs_for_reduce_dot_np = [X] + weights[:-1]

    second_to_last_layer = reduce(
        lambda a, b: relu_np(
            np.dot(
                a,
                b,
            )
        ),
        matrixs_for_reduce_dot_np,
    )

    return np.exp(np.dot(second_to_last_layer, weights[-1]))
