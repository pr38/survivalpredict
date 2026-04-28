import numpy as np
import pytest

from ..estimators import (
    AalenAdditiveHazard,
    ParametricDiscreteTimePH,
    CoxElasticNetPH,
    CoxProportionalHazard,
    CoxNNetPH,
    KaplanMeierSurvivalEstimator,
)
from ..metrics import brier_scores_administrative, integrated_brier_score_administrative
from ..validation import sur_cross_val_score

times_start = np.array([0, 1, 2, 3, 0, 1, 2, 3])
times = np.array([1, 2, 3, 4, 1, 2, 3, 4])
events = np.array([False, False, False, True, False, False, True, True])
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


@pytest.mark.parametrize(
    "estimator_class",
    [
        AalenAdditiveHazard,
        ParametricDiscreteTimePH,
        CoxElasticNetPH,
        CoxProportionalHazard,
        CoxNNetPH,
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
