import numbers
import time
from typing import Callable, Literal, Optional

import numpy as np
from joblib.parallel import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import check_cv

from .estimators import SurvivalPredictBase
from .metrics import (
    _integrated_brier_score_administrative,
    _integrated_brier_score_ipcw,
)
from .utils import validate_survival_data


def _aggregate_score_dicts(scores):
    """taken from sklearn.model_selection._validation.py, with the intent of avoiding too much dependenace to sk's internal api"""
    return {
        key: (
            np.asarray([score[key] for score in scores])
            if isinstance(scores[0][key], numbers.Number)
            else [score[key] for score in scores]
        )
        for key in scores[0]
    }


def _sur_fit_and_score(
    estimator,
    X,
    times,
    events,
    scorer,
    train,
    test,
    verbose,
    parameters,
    score_params,
    return_train_score,
    return_estimator,
    return_times,
    return_parameters,
    return_n_test_samples,
    method,
    brier_score_max_time: Optional[int] = None,
    brier_score_average_by_time: Optional[bool] = True,
):

    start_time = time.time()

    if brier_score_max_time is None:
        brier_score_max_time = int(max(times))

    X_train = X[train]
    X_test = X[test]

    times_train = times[train]
    times_test = times[test]

    events_train = events[train]
    events_test = events[test]

    estimator.fit(X_train, times_train, events_train, **parameters)

    if type(method) == str:
        func = getattr(estimator, method)
        predictions = func(X_test)
        if return_train_score:
            train_predictions = func(X_train)

    elif isinstance(method, Callable):
        predictions = method(estimator, X_test)
        if return_train_score:
            train_predictions = method(estimator, X_train)

    elif issubclass(type(estimator), SurvivalPredictBase):
        predictions = estimator.predict(X_test, max_time=brier_score_max_time)

        if return_train_score:
            train_predictions = estimator.predict(
                X_train, max_time=brier_score_max_time
            )

    else:
        predictions = estimator.predict(X_test)

        if return_train_score:
            train_predictions = estimator.predict(X_train)

    fit_time = time.time() - start_time

    if scorer == "integrated_brier_score_ipcw":
        test_score = _integrated_brier_score_ipcw(
            predictions,
            events=events_test,
            times=times_test,
            events_for_ipcw=events_train,
            times_for_ipcw=times_train,
            max_time=brier_score_max_time,
            average_by_time=brier_score_average_by_time,
        )
        if return_train_score:
            train_scores = _integrated_brier_score_ipcw(
                train_predictions,
                events=events_train,
                times=times_train,
                max_time=brier_score_max_time,
                average_by_time=brier_score_average_by_time,
            )

    elif scorer == "integrated_brier_score_administrative":
        test_score = _integrated_brier_score_administrative(
            predictions,
            events_test,
            times_test,
            max_time=brier_score_max_time,
            average_by_time=brier_score_average_by_time,
        )
        if return_train_score:
            train_scores = _integrated_brier_score_administrative(
                train_predictions,
                events=events_train,
                times=times_train,
                max_time=brier_score_max_time,
                average_by_time=brier_score_average_by_time,
            )

    elif isinstance(scorer, Callable):
        test_score = scorer(predictions, events_test, times_test, **score_params)
        if return_train_score:
            train_scores = scorer(
                train_predictions, events_train, times_train, **score_params
            )
    else:
        raise ValueError("unknown scorer")

    score_time = time.time() - start_time - fit_time

    result = {}
    result["test_scores"] = test_score
    if return_train_score:
        result["train_scores"] = train_scores

    if return_n_test_samples:
        result["n_test_samples"] = X_test.shape[0]
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


def sur_cross_validate(
    estimator,
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    *,
    groups=None,
    scoring: Optional[
        Literal["integrated_brier_score_administrative", "integrated_brier_score_ipcw"]
    ] = None,
    cv: Optional[int] = None,
    n_jobs: Optional[int] = None,
    verbose=0,
    params=None,
    pre_dispatch="2*n_jobs",
    return_train_score: Optional[bool] = False,
    return_estimator: Optional[bool] = False,
    brier_score_max_time: Optional[int] = None,
    brier_score_average_by_time: Optional[bool] = True,
    return_parameters: Optional[bool] = None,
    return_n_test_samples: Optional[bool] = None,
    method: Optional[bool] = None,
):
    if return_train_score is None:
        return_train_score = False
    elif not isinstance(return_train_score, bool):
        raise ValueError("return_train_score should be boolian")

    if return_estimator is None:
        return_estimator = False
    elif not isinstance(return_estimator, bool):
        raise ValueError("return_estimator should be boolian")

    X, times, events = validate_survival_data(X, times, events)

    if scoring == None:
        scoring = "integrated_brier_score_administrative"
    elif scoring not in [
        "integrated_brier_score_administrative",
        "integrated_brier_score_ipcw",
    ]:
        raise ValueError(
            'Score must be "integrated_brier_score_administrative" or "integrated_brier_score_ipcw"'
        )

    cv = check_cv(cv)

    params = {} if params is None else params

    indices = cv.split(X, groups=groups)

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)

    results = parallel(
        delayed(_sur_fit_and_score)(
            clone(estimator),
            X,
            times,
            events,
            scorer=scoring,
            train=train,
            test=test,
            verbose=verbose,
            parameters=params,
            score_params=scoring,
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            return_parameters=return_parameters,
            return_n_test_samples=return_n_test_samples,
            method=method,
            brier_score_max_time=brier_score_max_time,
            brier_score_average_by_time=brier_score_average_by_time,
        )
        for train, test in indices
    )

    # return_parameters', 'return_n_test_samples', and 'method'

    return _aggregate_score_dicts(results)


def sur_cross_val_score(
    estimator,
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    *,
    groups=None,
    scoring: Optional[
        Literal["integrated_brier_score_administrative", "integrated_brier_score_ipcw"]
    ] = None,
    cv: Optional[int] = None,
    n_jobs: Optional[int] = None,
    verbose=0,
    params=None,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
    brier_score_max_time: Optional[int] = None,
    brier_score_average_by_time: Optional[bool] = True,
):

    cv_result = sur_cross_validate(
        estimator,
        X,
        times,
        events,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        params=params,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
        brier_score_max_time=brier_score_max_time,
        brier_score_average_by_time=brier_score_average_by_time,
    )

    return cv_result["test_scores"]
