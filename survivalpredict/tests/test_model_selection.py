import pytest

from ..estimators import CoxProportionalHazard
from ..model_selection import Sur_GridSearchCV
from ..datasets import load_iranian_telecom_churn


@pytest.fixture
def X_times_events():

    data = load_iranian_telecom_churn()

    X = data["X"]
    times = data["times"]
    events = data["events"]

    return X, times, events


def test_sur_gridsearch(X_times_events):
    X, times, events = X_times_events

    grid_search = {
        "ties": ["breslow", "efron"],
        "alpha": [0, 0.1, 1, 10, 50, 100],
        "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1],
    }

    gscv = Sur_GridSearchCV(
        CoxProportionalHazard(),
        param_grid=grid_search,
        # error_score="raise",
        brier_score_max_time=int(max(times)),
        n_jobs=1,
    )

    gscv.fit(X, times, events)
