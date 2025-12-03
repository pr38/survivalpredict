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

from .utils import validate_survival_data
from .validation import _aggregate_score_dicts, _sur_fit_and_score


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

    def _refit_best_estimator(self, X, times, events, fit_params,strata=None):

        self.best_estimator_ = clone(self.estimator).set_params(
            **clone(self.best_params_, safe=False)
        )

        refit_start_time = time.time()

        if strata is not None:
            self.best_estimator_.fit(X, times, events,strata=strata, **fit_params)
        else:
            self.best_estimator_.fit(X, times, events, **fit_params)

        refit_end_time = time.time()
        self.refit_time_ = refit_end_time - refit_start_time

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, times, events,strata=None, **params):

        X, times, events = validate_survival_data(X, times, events)

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
                strata=strata
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

        self._refit_best_estimator(X, times, events, params,strata=strata)

        return self


class Sur_GridSearchCV(Sur_BaseSearchCV):

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
        verbose=0,
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
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _get_parameters_to_search(self):
        return ParameterGrid(self.param_grid)


class Sur_RandomizedSearchCV(Sur_BaseSearchCV):
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
        verbose=0,
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
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _get_parameters_to_search(self):
        return ParameterSampler(
            self.param_distributions, self.n_iter, random_state=self.random_state
        )
