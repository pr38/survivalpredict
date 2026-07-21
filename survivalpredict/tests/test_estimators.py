from inspect import signature

import numpy as np
import pytest

from ..datasets import load_iranian_telecom_churn
from ..estimators import (
    AalenAdditiveHazard,
    CoxNeuralNetPH,
    CoxPHElasticNet,
    CoxProportionalHazard,
    KaplanMeierSurvivalEstimator,
    KNeighborsSurvival,
    ParametricDiscreteTimePH,
    MultiTaskLogisticRegression,
)

data = load_iranian_telecom_churn()

X = np.ascontiguousarray(data["X"])
times = data["times"].astype(np.int64)
events = data["events"].astype(np.bool_)
times_start = np.zeros(X.shape[0]).astype(np.int64)
strata = np.random.randint(0, 2, X.shape[0])


@pytest.mark.parametrize(
    "estimator",
    [
        CoxProportionalHazard,
        ParametricDiscreteTimePH,
        KaplanMeierSurvivalEstimator,
        KNeighborsSurvival,
        CoxNeuralNetPH,
        AalenAdditiveHazard,
        CoxPHElasticNet,
        MultiTaskLogisticRegression,
    ],
)
@pytest.mark.parametrize("check_input", [True, False])
@pytest.mark.parametrize("strata", [None, strata])
@pytest.mark.parametrize("times_start", [None, times_start])
def test_estimators(estimator, check_input, strata, times_start):
    est = estimator()

    fit_arg_keys = set(signature(estimator.fit).parameters.keys())

    kwargs = {}
    kwargs["check_input"] = check_input
    predict_kwargs = {}

    if "strata" in fit_arg_keys:
        kwargs["strata"] = strata
        predict_kwargs["strata"] = strata

    if "times_start" in fit_arg_keys:
        kwargs["times_start"] = times_start

    est.fit(X, times, events, **kwargs)

    prediction = est.predict(X, **predict_kwargs)

    hazard = est.predict_hazard(X, **predict_kwargs)
    survival_from_hazard = est.convert_hazard_to_survival(hazard)

    np.testing.assert_almost_equal(prediction, survival_from_hazard)
