import numpy as np
import pytensor.tensor as pt
import pymc as pm
import pymc_extras as pmx


def _scale_times(times, time_max):
    return times / time_max


def _weibull_pdf(self, x, a, v):
    return (v / a) * (x / a) ** (v - 1)


def _chen_pdf(x, c, k):
    e_x_k = np.exp(x**k)
    return c * k * x ** (k - 1) * e_x_k * np.exp(c - (c * e_x_k))


def get_parametric_discrete_time_ph_model(
    X, times, events, base_hazard_pdf_callable, max_time=None, labes_names=None
):
    if not max_time:
        max_time = times.max()

    if not labes_names:
        labes_names = np.arange(X.shape[1])

    times_of_intrest = np.arange(1, max_time + 1)
    times_of_intrest_norm = _scale_times(times_of_intrest, max_time)

    survived_at_times = times[:, np.newaxis] >= times_of_intrest
    not_censored_at_times = np.logical_or(
        survived_at_times, events.astype(np.bool_)[:, np.newaxis]
    )

    row_ids = np.arange(X.shape[0])

    with pm.Model(
        coords={"labes": labes_names, "times": times_of_intrest, "row_ids": row_ids}
    ) as model:

        data = pm.Data("data", X, dims=("row_ids", "labes"))

        coefs = pm.Normal("coefs", sigma=50, dims="labes")

        a = pm.HalfNormal("a", 5)
        b = pm.HalfNormal("b", 5)

        base_hazards = base_hazard_pdf_callable(times_of_intrest_norm, a, b)

        relative_risk = pt.exp(pt.dot(data, coefs))

        hazard = base_hazards * relative_risk[:, None]
        hazard_estimation_cumsum = pt.cumsum(hazard, axis=1)

        survival = pt.exp(-hazard_estimation_cumsum)

        def censored_bernoulli_logp(value, p, noncensored_mask):
            dis = pm.Bernoulli.logp(value, p=p)
            return dis.flatten()[noncensored_mask.flatten()]

        y = pm.DensityDist(
            "y",
            survival,
            not_censored_at_times,
            logp=censored_bernoulli_logp,
            observed=survived_at_times,
        )

    return model


def train_parametric_discrete_time_ph_model(
    X, times, events, hazard_pdf_callable
) -> tuple[np.array, float, float]:

    model = get_parametric_discrete_time_ph_model(X, times, events, hazard_pdf_callable)

    with model:
        mle = pmx.find_MAP(
            compile_kwargs={"mode": "JAX"},
            progressbar=False,
        )

    coefs = mle.posterior["coefs"].values.flatten()
    a = mle.posterior["a"].item()
    b = mle.posterior["b"].item()

    return coefs, a, b


def predict_parametric_discrete_time_ph_model(
    X, coefs, a, b, max_time, base_hazard_pdf_callable
):

    times_of_intrest = np.arange(1, max_time + 1)
    times_of_intrest_norm = _scale_times(times_of_intrest, max_time)

    base_hazards = base_hazard_pdf_callable(times_of_intrest_norm, a, b)

    relative_risk = np.exp(np.dot(X, coefs))

    hazard = base_hazards * relative_risk[:, None]
    hazard_estimation_cumsum = np.cumsum(hazard, axis=1)

    survival = np.exp(-hazard_estimation_cumsum)

    return survival
