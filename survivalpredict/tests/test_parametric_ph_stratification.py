import pytest
import numpy as np

from ..estimators import ParametricDiscreteTimePH
from ..validation import sur_cross_val_score
from ..datasets import load_iranian_telecom_churn


@pytest.fixture
def X_times_events_strata():
    iranian_telecom_churn = load_iranian_telecom_churn()

    X_raw = iranian_telecom_churn["X"]

    times = iranian_telecom_churn["times"]
    events = iranian_telecom_churn["events"]

    position_of_age_col = int(
        np.argwhere(iranian_telecom_churn["column_names"] == "age")[0][0]
    )
    age = X_raw[:, position_of_age_col]
    bins = np.quantile(age, [0.25, 0.5, 0.75])

    strata = np.digitize(age, bins=bins).flatten()
    X = X_raw[:, ~np.isin(range(X_raw.shape[1]), position_of_age_col)]

    def normalize(X):
        return np.subtract(X, X.mean(axis=0)) / X.std(axis=0)

    X = normalize(X)

    return X, times, events, strata


def test_ParametricDiscreteTimePH(X_times_events_strata):

    X, times, events, strata = X_times_events_strata

    pph = ParametricDiscreteTimePH(pytensor_mode="JAX", distribution="chen")
    pph.fit(X, times, events, strata=strata)

    assert (
        bool((pph.coef_ == 0).all()) == False
    )  # model was trained, and coefs are not zero

    n_strata = len(np.unique(strata))

    assert pph.base_hazard_prams_.shape == (
        n_strata,
        2,
    )  # chen distribution has 2 parameters

    predictions = pph.predict(X, strata=strata)

    max_time = int(max(times))

    assert predictions.shape == (X.shape[0], max_time)

    scores = sur_cross_val_score(
        ParametricDiscreteTimePH(pytensor_mode="JAX"), X, times, events, strata=strata
    )

    assert all(np.isnan(scores)) == False
