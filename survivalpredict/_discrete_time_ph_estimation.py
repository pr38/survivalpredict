import numpy as np
import pytensor.tensor as pt
import pymc as pm
import pymc_extras as pmx


def _scale_times(times, time_max):
    return times / time_max


def _weibull_pdf(x, prams):
    return (prams[0] / prams[1]) * (x / prams[1]) ** (prams[0] - 1)


def _chen_pdf(x, prams):
    e_x_k = np.exp(x ** prams[1])
    return (
        prams[0]
        * prams[1]
        * x ** (prams[1] - 1)
        * e_x_k
        * np.exp(prams[0] - (prams[0] * e_x_k))
    )


def get_parametric_discrete_time_ph_model(
    X,
    times,
    events,
    base_hazard_pdf_callable,
    n_base_hazard_prams,
    max_time=None,
    labes_names=None,
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
    base_hazard_prams_ids = range(n_base_hazard_prams)

    with pm.Model(
        coords={
            "labes": labes_names,
            "times": times_of_intrest,
            "row_ids": row_ids,
            "base_hazard_prams_ids": base_hazard_prams_ids,
        }
    ) as model:

        data = pm.Data("data", X, dims=("row_ids", "labes"))

        coefs = pm.Normal("coefs", sigma=50, dims="labes")

        base_hazard_prams = pm.Exponential(
            "base_hazard_prams", 5, dims="base_hazard_prams_ids"
        )

        base_hazards = base_hazard_pdf_callable(
            times_of_intrest_norm, base_hazard_prams
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

    return model


def train_parametric_discrete_time_ph_model(
    X, times, events, hazard_pdf_callable, n_base_hazard_prams
) -> tuple[np.array, np.array]:

    model = get_parametric_discrete_time_ph_model(
        X, times, events, hazard_pdf_callable, n_base_hazard_prams
    )

    with model:
        mle = pmx.find_MAP(
            compile_kwargs={"mode": "JAX"},
            progressbar=False,
        )

    coefs = mle.posterior["coefs"].values.flatten()
    base_hazard_prams = mle.posterior["base_hazard_prams"].values.flatten()

    return coefs, base_hazard_prams


def predict_parametric_discrete_time_ph_model(
    X, coefs, base_hazard_prams, max_time, base_hazard_pdf_callable
):

    times_of_intrest = np.arange(1, max_time + 1)
    times_of_intrest_norm = _scale_times(times_of_intrest, max_time)

    base_hazards = base_hazard_pdf_callable(times_of_intrest_norm, base_hazard_prams)

    relative_risk = np.exp(np.dot(X, coefs))

    hazard = base_hazards * relative_risk[:, None]
    hazard_estimation_cumsum = np.cumsum(hazard, axis=1)

    survival = np.exp(-hazard_estimation_cumsum)

    return survival
