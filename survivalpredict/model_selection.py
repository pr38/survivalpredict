import numbers
import time
from abc import ABCMeta, abstractmethod
from itertools import product
from typing import Any, Callable, Literal, Optional

import numpy as np
from joblib.parallel import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import ParameterGrid, ParameterSampler
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions

from ._data_validation import validate_survival_data, validate_times_start_array
from .validation import _aggregate_score_dicts, _sur_fit_and_score

__all__ = ["Sur_GridSearchCV", "Sur_RandomizedSearchCV"]


class Sur_BaseSearchCV(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "scoring": [
            StrOptions(
                {"integrated_brier_score_administrative", "integrated_brier_score_ipcw"}
            ),
            callable,
            None,
        ],
        "brier_score_max_time": [numbers.Integral, None],
        "n_jobs": [numbers.Integral, None],
        "refit": ["boolean"],
        "cv": ["cv_object"],
        "verbose": ["verbose"],
        "pre_dispatch": [numbers.Integral, str],
        "error_score": [StrOptions({"raise"}), numbers.Real],
        "return_train_score": ["boolean"],
    }

    def __init__(
        self,
        estimator,
        param_grid: dict | list,
        *,
        brier_score_max_time: Optional[int] = None,
        scoring: Optional[
            Literal[
                "integrated_brier_score_administrative", "integrated_brier_score_ipcw"
            ]
            | Callable
        ] = None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=True,
    ):
        self.scoring = scoring
        self.brier_score_max_time = brier_score_max_time
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    def _process_results(
        self, _sur_fit_and_score_results: list[dict], n_candidates: int, n_splits: int
    ):

        agg_results = _aggregate_score_dicts(_sur_fit_and_score_results)

        results = []

        for row in agg_results["parameters"][::n_splits]:
            new_row = {"params": row}
            for key, value in row.items():
                new_row[f"param_{key}"] = value
            results.append(new_row)

        test_scores = agg_results["test_scores"].reshape(n_candidates, n_splits)
        fit_time = agg_results["fit_time"].reshape(n_candidates, n_splits)

        mean_fit_time = np.average(fit_time, axis=1)
        std_fit_time = np.std(fit_time, axis=1)

        mean_test_scores = np.average(test_scores, axis=1)
        std_test_scores = np.std(test_scores, axis=1)

        rank_test_score = rankdata(mean_test_scores, method="min").astype(
            np.int32, copy=False
        )

        return_train_score = self.return_train_score

        if return_train_score:
            train_scores = agg_results["train_scores"].reshape(n_candidates, n_splits)
            mean_train_scores = np.average(train_scores, axis=1)
            std_train_scores = np.std(train_scores, axis=1)

        for i in range(n_candidates):
            row = results[i]

            row["rank_test_score"] = rank_test_score[i]
            row["mean_fit_time"] = mean_fit_time[i]
            row["std_fit_time"] = std_fit_time[i]

            row["mean_test_scores"] = mean_test_scores[i]
            row["std_test_scores"] = std_test_scores[i]

            if return_train_score:
                row["mean_train_scores"] = mean_train_scores[i]
                row["std_train_scores"] = std_train_scores[i]

            test_scores_i = test_scores[i]

            for ii in range(n_splits):
                row[f"split{ii}_test_score"] = test_scores_i[ii]

        self.cv_results_ = results

        self.best_index_ = rank_test_score.argmin()
        self.best_params_ = results[self.best_index_]["params"]
        self.best_score_ = mean_test_scores[self.best_index_]

    def _refit_best_estimator(self, X, times, events, fit_params, strata=None):

        self.best_estimator_ = clone(self.estimator).set_params(
            **clone(self.best_params_, safe=False)
        )

        refit_start_time = time.time()

        if strata is not None:
            self.best_estimator_.fit(X, times, events, strata=strata, **fit_params)
        else:
            self.best_estimator_.fit(X, times, events, **fit_params)

        refit_end_time = time.time()
        self.refit_time_ = refit_end_time - refit_start_time

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, times, events, strata=None, times_start=None):
        """
        Run fit with sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        times : array-like of shape (n_samples), dtype=np.int64
            Point in time last observed.

        events : array-like of shape (n_samples), dtype=np.bool_
            Experianed event.

        strata : array-like of shape (n_samples,), dtype=np.int64, default=None
            If passed in, associated strata for per observation.

        times_start : array-like of shape (n_samples, dtype=np.int64), default=None
            Starting point for observation. If not passed in, all times_start
            times are assumed to be 0.

        Returns
        -------
        object
            Instance of fitted estimator.
        """

        X, times, events = validate_survival_data(X, times, events)

        times_start = validate_times_start_array(times_start, times)

        if strata is not None:
            self._uses_strata = True
        else:
            self._uses_strata = False

        if self.scoring == None:
            scoring = "integrated_brier_score_administrative"
        else:
            scoring = self.scoring

        cv = check_cv(self.cv)

        parameters_to_search = self._get_parameters_to_search()
        n_candidates = len(parameters_to_search)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        indices = cv.split(X)
        n_splits = cv.get_n_splits()

        if self.brier_score_max_time == None:
            brier_score_max_time = int(max(times))
        else:
            brier_score_max_time = self.brier_score_max_time

        results = parallel(
            delayed(_sur_fit_and_score)(
                clone(base_estimator),
                X,
                times,
                events,
                scorer=scoring,
                train=train,
                test=test,
                verbose=self.verbose,
                parameters=parameters,
                fit_params={},
                score_params={},
                return_train_score=self.return_train_score,
                return_times=True,
                return_estimator=False,
                return_parameters=True,
                return_n_test_samples=False,
                method=False,
                brier_score_max_time=brier_score_max_time,
                error_score=self.error_score,
                strata=strata,
                times_start=times_start,
            )
            for parameters, (train, test) in product(
                parameters_to_search,
                indices,
            )
        )

        if len(results) < 1:
            raise ValueError("No fits where preformed")

        elif len(results) != n_candidates * n_splits:
            raise ValueError("Unexpcted number of results")

        self._process_results(results, n_candidates, n_splits)

        self._refit_best_estimator(X, times, events, times_start=times_start, strata=strata)

        return self


class Sur_GridSearchCV(Sur_BaseSearchCV):
    """
    Survivalpredict's native exhaustive search over specified parameter values
    for an estimator.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid. Like scikit-learn,
    parallelism is achieved with joblib.

    Parameters
    ----------

    estimator : estimator object
        Survivalpredict native estimator.

    param_grid : dict | list
        Dictionary with parameters names (``str``) as keys and lists of
        parameter settings to try as values, or a list of such dictionaries, in
        which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any sequence of parameter
        settings.

    brier_score_max_time : Optional[int], default=None
        Maximum time to evaluate survival curves for brier scores. If None,
        will evaluate all times seen.

    scoring : Optional[Literal["integrated_brier_score_administrative", "integrated_brier_score_ipcw"] | Callable], default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

    n_jobs : int, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. Fully compatible
        with scikit-learn CV components.

        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an explosion of
        memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:

        - None, in which case all the jobs are immediately created and spawned. Use
          this for lightweight and fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs
        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in '2*n_jobs'

    pre_dispatch : int, or str, default="2*n_jobs"
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an explosion of
        memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:

        - None, in which case all the jobs are immediately created and spawned. Use
          this for lightweight and fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs
        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores. Computing training scores is used to get insights on how
        different parameter settings impact the overfitting/underfitting
        trade-off. However computing the scores on the training set can be
        computationally expensive and is not strictly required to select the
        parameters that yield the best generalization performance.
    """

    _parameter_constraints: dict = {
        **Sur_BaseSearchCV._parameter_constraints,
        "param_grid": [dict, list],
    }

    def __init__(
        self,
        estimator,
        param_grid: dict | list,
        *,
        brier_score_max_time: Optional[int] = None,
        scoring: Optional[
            Literal[
                "integrated_brier_score_administrative", "integrated_brier_score_ipcw"
            ]
            | Callable
        ] = None,
        n_jobs=None,
        refit=True,
        cv=None,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):

        self.param_grid = param_grid

        super().__init__(
            scoring=scoring,
            brier_score_max_time=brier_score_max_time,
            estimator=estimator,
            param_grid=param_grid,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=0,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _get_parameters_to_search(self):
        return ParameterGrid(self.param_grid)


class Sur_RandomizedSearchCV(Sur_BaseSearchCV):
    """
    Survivalpredict's native randomized search on hyper parameters.

    Not all parameter values are tried out, but rather a fixed number of
    parameter settings is sampled from the specified distributions. The number
    of parameter settings that are tried is given by n_iter. Like scikit-learn,
    parallelism is achieved with joblib.

    Parameters
    ----------
    estimator : estimator object
        Survivalpredict native estimator.

    param_distributions : dict | list
        Dictionary with parameters names (str) as keys and distributions or
        lists of parameters to try. Distributions must provide a rvs method for
        sampling (such as those from scipy.stats.distributions). If a list is
        given, it is sampled uniformly. If a list of dicts is given, first a
        dict is sampled uniformly, and then a parameter is sampled using that
        dict as above.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades off
        runtime vs quality of the solution.

    brier_score_max_time : Optional[int], default=None
        Maximum time to evaluate survival curves for brier scores. If None,
        will evaluate all times seen.

    scoring : Optional[Literal["integrated_brier_score_administrative", "integrated_brier_score_ipcw"] | Callable], default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

    n_jobs : int, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. Fully compatible
        with scikit-learn CV components.

        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an explosion of
        memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:

        - None, in which case all the jobs are immediately created and spawned. Use
          this for lightweight and fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs
        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in '2*n_jobs'

    pre_dispatch : int, or str, default="2*n_jobs"
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an explosion of
        memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:

        - None, in which case all the jobs are immediately created and spawned. Use
          this for lightweight and fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs
        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in '2*n_jobs'

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple function calls.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores. Computing training scores is used to get insights on how
        different parameter settings impact the overfitting/underfitting
        trade-off. However computing the scores on the training set can be
        computationally expensive and is not strictly required to select the
        parameters that yield the best generalization performance.
    """

    _parameter_constraints: dict = {
        **Sur_BaseSearchCV._parameter_constraints,
        "param_distributions": [dict, list],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator,
        param_distributions: dict | list,
        *,
        n_iter=10,
        brier_score_max_time: Optional[int] = None,
        scoring: Optional[
            Literal[
                "integrated_brier_score_administrative", "integrated_brier_score_ipcw"
            ]
            | Callable
        ] = None,
        n_jobs=None,
        refit=True,
        cv=None,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):

        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

        super().__init__(
            scoring=scoring,
            brier_score_max_time=brier_score_max_time,
            estimator=estimator,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=0,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _get_parameters_to_search(self):
        return ParameterSampler(
            self.param_distributions, self.n_iter, random_state=self.random_state
        )
