import numbers
import time
from typing import Callable, Literal, Optional, Union

import numpy as np
from joblib.parallel import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import check_cv

from .estimators import _SurvivalPredictBase
from .metrics import (
    _integrated_brier_score_administrative,
    _integrated_brier_score_ipcw,
)
from ._data_validation import validate_survival_data, _as_int_np_array, _as_int

__all__ = ["sur_cross_val_score", "sur_cross_validate"]


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
    parameters: dict,
    fit_params: dict,
    score_params: dict,
    return_train_score: Optional[bool],
    return_estimator: Optional[bool],
    return_times: Optional[bool],
    return_parameters: Optional[bool],
    return_n_test_samples: Optional[bool],
    method: Optional[Union[str, Callable]],
    strata: Optional[np.array] = None,
    brier_score_max_time: Optional[int] = None,
    error_score: numbers.Real | Literal["raise"] = "raise",
    times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
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

    has_strata = strata is not None
    is_left_censored = times_start is not None

    fit_params = fit_params.copy()
    predict_prams = {}

    if has_strata:
        strata_train = strata[train]
        strata_test = strata[test]
        fit_params["strata"] = strata_train
        predict_prams["strata"] = strata_test

    if is_left_censored:
        times_start_train = times_start[train]
        times_start_test = times_start[test]
        fit_params["times_start"] = times_start_train
    else:
        times_start_train = None
        times_start_test = None

    if parameters:
        estimator = estimator.set_params(**clone(parameters, safe=False))

    try:
        estimator.fit(X_train, times_train, events_train, **fit_params)
        fit_time = time.time() - start_time

    except:

        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            fit_time = time.time() - start_time
            test_score = error_score
            train_scores = error_score
            score_time = float("nan")

    else:

        if type(method) == str:
            func = getattr(estimator, method)
            predictions = func(X_test)
            if return_train_score:
                train_predictions = func(X_train)

        elif isinstance(method, Callable):
            predictions = method(estimator, X_test)
            if return_train_score:
                train_predictions = method(estimator, X_train)

        elif issubclass(type(estimator), _SurvivalPredictBase):
            predictions = estimator.predict(
                X_test, max_time=brier_score_max_time, **predict_prams
            )

            if return_train_score:
                train_predictions = estimator.predict(
                    X_train, max_time=brier_score_max_time, **predict_prams
                )

        else:
            predictions = estimator.predict(X_test)

            if return_train_score:
                train_predictions = estimator.predict(X_train)

        if scorer == "integrated_brier_score_ipcw":
            test_score = _integrated_brier_score_ipcw(
                predictions,
                events=events_test,
                times=times_test,
                events_for_ipcw=events_train,
                times_for_ipcw=times_train,
                max_time=brier_score_max_time,
            )
            if return_train_score:
                train_scores = _integrated_brier_score_ipcw(
                    train_predictions,
                    events=events_train,
                    times=times_train,
                    max_time=brier_score_max_time,
                )

        elif scorer == "integrated_brier_score_administrative":
            test_score = _integrated_brier_score_administrative(
                predictions,
                events_test,
                times_test,
                max_time=brier_score_max_time,
                times_start=times_start_test,
            )
            if return_train_score:
                train_scores = _integrated_brier_score_administrative(
                    train_predictions,
                    events=events_train,
                    times=times_train,
                    max_time=brier_score_max_time,
                    times_start=times_start_train,
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
    strata: Optional[np.ndarray] = None,
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
    return_parameters: Optional[bool] = None,
    return_n_test_samples: Optional[bool] = None,
    method: Optional[bool] = None,
    error_score: Literal["raise"] | np.number | float = np.nan,
    times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
):
    """
    Evaluate survival metrics by cross-validation and also record fit/score times.

    Parameters
    ----------
    estimator : survivalpredict estimator
        Instance of a survivalpredict compatible estimator.

    X : array-like of shape (n_samples, n_features)
        Training data.

    times : array-like of shape (n_samples), dtype=np.int64
        Point in time last observed.

    events : array-like of shape (n_samples), dtype=np.bool_
        Experianed event.

    strata : array-like of shape (n_samples,), dtype=np.int64, default=None
        If passed in, associated strata for per observation.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into train/test set.
        Only used in conjunction with a “Group” cv instance (e.g., GroupKFold).

    scoring : {"integrated_brier_score_administrative", "integrated_brier_score_ipcw"} , default="integrated_brier_score_administrative"
        survival metrics

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    params : dict, default=None
        Parameters to pass to the underlying estimator's ``fit``, the scorer,
        and the CV splitter.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in '2*n_jobs'

    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.

    brier_score_max_time : int, default=None
        Maximum time to evaluate survival curves. If None, will evaluate all
        times seen.

    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.

    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.

    method : str, default='predict'
       Method used to predict for estimator.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    times_start : array-like of shape (n_samples, dtype=np.int64), default=None
        Starting point for observation. If not passed in, all times_start
        times are assumed to be 0.

    Returns
    -------
    dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.
    """
    if return_train_score is None:
        return_train_score = False
    elif not isinstance(return_train_score, bool):
        raise ValueError("return_train_score should be boolean")

    if return_estimator is None:
        return_estimator = False
    elif not isinstance(return_estimator, bool):
        raise ValueError("return_estimator should be boolean")

    X, times, events = validate_survival_data(X, times, events)

    if brier_score_max_time is not None:
        brier_score_max_time = _as_int(brier_score_max_time, "max_time")

    if strata is not None:
        strata = _as_int_np_array(strata)

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
            parameters=None,
            fit_params=params,
            score_params={},
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            return_parameters=return_parameters,
            return_n_test_samples=return_n_test_samples,
            method=method,
            brier_score_max_time=brier_score_max_time,
            error_score=error_score,
            strata=strata,
            times_start=times_start,
        )
        for train, test in indices
    )

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
    error_score: Literal["raise"] | np.number | float = np.nan,
    brier_score_max_time: Optional[int] = None,
    method: Optional[bool] = None,
    strata: Optional[np.array] = None,
    times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
):
    """
    Evaluate survival score by cross-validation.

    Parameters
    ----------

    estimator : survivalpredict estimator
        Instance of a survivalpredict compatible estimator.

    X : array-like of shape (n_samples, n_features)
        Training data.

    times : array-like of shape (n_samples), dtype=np.int64
        Point in time last observed.

    events : array-like of shape (n_samples), dtype=np.bool_
        Experianed event.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into train/test set.
        Only used in conjunction with a “Group” cv instance (e.g., GroupKFold).

    scoring : {"integrated_brier_score_administrative", "integrated_brier_score_ipcw"} , default="integrated_brier_score_administrative"
        survival metrics

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    params : dict, default=None
        Parameters to pass to the underlying estimator's ``fit``, the scorer,
        and the CV splitter.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    brier_score_max_time : int, default=None
        Maximum time to evaluate survival curves. If None, will evaluate all
        times seen.

    method : str, default='predict'
       Method used to predict for estimator.

    strata : array-like of shape (n_samples,), dtype=np.int64, default=None
        If passed in, associated strata for per observation.

    times_start : array-like of shape (n_samples, dtype=np.int64), default=None
        Starting point for observation. If not passed in, all times_start
        times are assumed to be 0.

    Returns
    -------
    ndarray of float of shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    """

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
        method=method,
        strata=strata,
        times_start=times_start,
    )

    return cv_result["test_scores"]
