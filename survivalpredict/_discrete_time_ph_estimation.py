from typing import Any, Callable, Literal, Optional
import warnings

import numpy as np
import pymc as pm  # type: ignore
import pymc_extras as pmx  # type: ignore
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable


def _scale_times(
    times: np.ndarray[tuple[int], np.dtype[np.int64]], time_max: int
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    return np.divide(times, time_max)  # type: ignore


def _weibull_pdf(x, params):
    return (params[0] / params[1]) * (x / params[1]) ** (params[0] - 1)


def _chen_pdf(x, params):
    e_x_k = np.exp(x ** params[1])
    return (
        params[0]
        * params[1]
        * x ** (params[1] - 1)
        * e_x_k
        * np.exp(params[0] - (params[0] * e_x_k))
    )


def _log_normal_pdf(x, params):
    return (
        1
        / (x * params[0] * np.sqrt(2 * params[1]))
        * np.exp(-((np.log(x) - params[1]) ** 2) / (2 * params[0] ** 2))
    )


def _log_logistic_pdf(x, params):
    return ((params[1] / params[0]) * (x / params[0]) ** (params[1] - 1)) / (
        (1 + (x / params[0]) ** params[1]) ** 2
    )


def _gompertz_pdf(x, params):
    return (
        params[0]
        * params[1]
        * np.exp(params[0] + params[1] * x - (params[0] * np.exp(params[1] * x)))
    )


def _additive_chen_weibull_pdf(x, params):
    return params[0] * params[1] * x ** (params[1] - 1) + params[2] * params[3] * (
        params[2] * x
    ) ** (params[3] - 1)


def get_parametric_discrete_time_ph_model(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    base_hazard_pdf_callable: Callable[
        [
            np.ndarray[tuple[int], np.dtype[np.float64]],
            TensorVariable,  # 1d float64
        ],
        TensorVariable,  # 1d float64
    ],
    n_base_hazard_params: int,
    max_time: Optional[int] = None,
    labes_names: list[str] | np.ndarray[tuple[int], np.dtype[Any]] | None = None,
    alpha: float = 0.0,
    l1_ratio: float = 0.5,
) -> pm.Model:

    if max_time is None:
        max_time = times.max()

    if labes_names is None:
        labes_names = np.arange(X.shape[1])

    times_of_intrest = np.arange(1, max_time + 1)
    times_of_intrest_norm = _scale_times(times_of_intrest, max_time)

    survived_at_times = times[:, np.newaxis] >= times_of_intrest
    not_censored_at_times = np.logical_or(
        survived_at_times, events.astype(np.bool_)[:, np.newaxis]
    )

    row_ids = np.arange(X.shape[0])
    base_hazard_params_ids = range(n_base_hazard_params)

    with pm.Model(
        coords={
            "labes": labes_names,
            "times": times_of_intrest,
            "row_ids": row_ids,
            "base_hazard_params_ids": base_hazard_params_ids,
        }
    ) as model:

        data = pm.Data("data", X, dims=("row_ids", "labes"))

        coefs = pm.Normal("coefs", sigma=50, dims="labes")

        base_hazard_params = pm.Exponential(
            "base_hazard_params", 5, dims="base_hazard_params_ids"
        )

        base_hazards = base_hazard_pdf_callable(
            times_of_intrest_norm, base_hazard_params
        )

        relative_risk = pt.exp(pt.dot(data, coefs))

        hazard = base_hazards * relative_risk[:, None]
        hazard_estimation_cumsum = pt.cumsum(hazard, axis=1)

        survival = pt.exp(-hazard_estimation_cumsum)

        def censored_bernoulli_logp(value, p, noncensored_mask):
            dis = pm.Bernoulli.logp(value, p=p)
            # return dis.flatten()[noncensored_mask.flatten()]
            return dis * noncensored_mask

        y = pm.DensityDist(
            "y",
            survival,
            not_censored_at_times,
            logp=censored_bernoulli_logp,
            observed=survived_at_times,
        )

        if alpha > 0.0:  # add elastic net loss
            l1 = alpha * l1_ratio * pt.abs(coefs).sum()
            l2 = 0.5 * alpha * (1.0 - l1_ratio) * pt.square(coefs).sum()
            pm.Potential("elasticnet_loss", -(l1 + l2))

    return model


def train_parametric_discrete_time_ph_model(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    hazard_pdf_callable: Callable[
        [
            np.ndarray[tuple[int], np.dtype[np.float64]],
            TensorVariable,  # 1d float64
        ],
        TensorVariable,  # 1d float64
    ],
    n_base_hazard_params: int,
    alpha: float,
    l1_ratio: float,
    pytensor_mode: Literal["JAX", "NUMBA"],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:

    model = get_parametric_discrete_time_ph_model(
        X,
        times,
        events,
        hazard_pdf_callable,
        n_base_hazard_params,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=UserWarning)

        with model:
            mle = pmx.find_MAP(
                compile_kwargs={"mode": pytensor_mode},
                progressbar=False,
            )

    coefs = mle.posterior["coefs"].values.flatten()
    base_hazard_params = mle.posterior["base_hazard_params"].values.flatten()

    return coefs, base_hazard_params


def predict_parametric_discrete_time_ph_model(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    coefs: np.ndarray[tuple[int], np.dtype[np.float64]],
    base_hazard_params: np.ndarray[tuple[int], np.dtype[np.float64]],
    max_time: int,
    base_hazard_pdf_callable: Callable[
        [
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        np.ndarray[tuple[int], np.dtype[np.float64]],
    ],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:

    times_of_intrest = np.arange(1, max_time + 1)
    times_of_intrest_norm = _scale_times(times_of_intrest, max_time)

    base_hazards = base_hazard_pdf_callable(times_of_intrest_norm, base_hazard_params)

    relative_risk = np.exp(np.dot(X, coefs))

    hazard = base_hazards * relative_risk[:, None]
    hazard_estimation_cumsum = np.cumsum(hazard, axis=1)

    survival = np.exp(-hazard_estimation_cumsum)

    return survival
