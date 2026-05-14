import numpy as np
import pytest

from ..estimators import (
    AalenAdditiveHazard,
    ParametricDiscreteTimePH,
    CoxPHElasticNet,
    CoxProportionalHazard,
    CoxNeuralNetPH,
    KaplanMeierSurvivalEstimator,
)
from ..metrics import brier_scores_administrative, integrated_brier_score_administrative
from ..validation import sur_cross_val_score

times_start = np.repeat(np.array([0, 1, 2, 3, 0, 1, 2, 3]), 3)
times = np.repeat(np.array([1, 2, 3, 4, 1, 2, 3, 10]), 3)
events = np.repeat(np.array([True, False, True, True, False, False, True, True]), 3)
X = np.array(
    [
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [2, 1, 0, 0],
        [3, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [2, 0, 1, 1],
        [3, 1, 1, 1],
    ]
)
X = np.concatenate([X] * 3)


@pytest.mark.parametrize(
    "estimator_class",
    [
        AalenAdditiveHazard,
        ParametricDiscreteTimePH,
        CoxPHElasticNet,
        CoxProportionalHazard,
        CoxNeuralNetPH,
        KaplanMeierSurvivalEstimator,
    ],
)
def test_left_censorship(estimator_class):
    est = estimator_class()
    est.fit(X, times, events, times_start=times_start)

    pred = est.predict(X)

    scores = brier_scores_administrative(pred, times, events, times_start=times_start)

    assert np.isnan(scores).all() == False

    cv_scores = sur_cross_val_score(
        estimator_class(), X, times, events, times_start=times_start, cv=2
    )

    assert np.isnan(cv_scores).all() == False
