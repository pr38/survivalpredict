import itertools
import warnings
from numbers import Integral, Real
from typing import Any, Callable, Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, _fit_context
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors._base import VALID_METRICS as VALID_METRICS_KNN
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ._base_hazard import _get_breslow_base_hazard
from ._cox_ph_estimation import train_cox_ph_breslow, train_cox_ph_efron
from ._data_validation import (
    _as_int,
    _as_int_np_array,
    _as_numeric_np_array,
    validate_survival_data,
)
from ._discrete_time_ph_estimation import (
    _additive_chen_weibull_pdf,
    _chen_pdf,
    _gompertz_pdf,
    _log_logistic_pdf,
    _log_normal_pdf,
    _scale_times,
    _weibull_pdf,
    get_parametric_discrete_time_ph_model,
    predict_parametric_discrete_time_ph_model,
    train_parametric_discrete_time_ph_model,
)
from ._neighbors import build_kaplan_meier_survival_curve_from_neighbors_indexes
from ._nonparametric import get_kaplan_meier_survival_curve_from_time_as_int_
from ._stratification import (
    get_l_div_m_stata_per_strata,
    map_new_strata,
    preprocess_data_for_cox_ph,
    split_and_preprocess_data_by_strata,
)

__all__ = [
    "CoxProportionalHazard",
    "ParametricDiscreteTimePH",
    "KaplanMeierSurvivalEstimator",
    "KNeighborsSurvival",
]


class _SurvivalPredictBase(BaseEstimator):
    def fit_predict(self, *args, **kwargs):
        predict_kwargs = kwargs.copy()

        if "max_time" in kwargs:
            del kwargs["max_time"]

        X = args[0]

        return self.fit(*args, **kwargs).predict(X, **predict_kwargs)


class CoxProportionalHazard(_SurvivalPredictBase):

    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "ties": [StrOptions({"breslow", "efron"})],
        "tol": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        *,
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        max_iter: Optional[int] = 100,
        ties: Optional[Literal["breslow", "efron"]] = "breslow",
        tol: float = 1e-9,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.ties = ties
        self.tol = tol

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, times, events, strata=None, weights=None, check_input=True):

        if check_input:
            X, times, events = validate_survival_data(X, times, events)

            if strata is not None:
                strata = _as_int_np_array(strata)

        self._max_time_observed = np.max(times)

        if hasattr(self, "coef_"):
            coefs = self.coef_
        else:
            coefs = np.zeros(X.shape[1])

        if self.ties == "efron":
            # get_l_div_m_stata_per_strata assumes sorted data
            argsort = times.argsort(kind="mergesort")
            times = times[argsort]
            events = events[argsort]
            X = X[argsort]

        (
            n_strata,
            seen_strata,
            X_strata,
            times_strata,
            events_strata,
            time_return_inverse_strata,
            n_unique_times_strata,
            event_counts_at_times_strata,
        ) = preprocess_data_for_cox_ph(X, times, events, strata)

        if self.ties == "breslow":

            coefs, loss = train_cox_ph_breslow(
                X_strata,
                events_strata,
                n_unique_times_strata,
                event_counts_at_times_strata,
                time_return_inverse_strata,
                n_strata,
                self.alpha,
                self.l1_ratio,
                coefs,
                self.max_iter,
                self.tol,
            )

        elif self.ties == "efron":
            l_div_m_stata = get_l_div_m_stata_per_strata(
                events_strata,
                times_strata,
                time_return_inverse_strata,
                n_unique_times_strata,
            )

            coefs, loss = train_cox_ph_efron(
                n_strata,
                X_strata,
                events_strata,
                n_unique_times_strata,
                l_div_m_stata,
                time_return_inverse_strata,
                self.alpha,
                self.l1_ratio,
                coefs,
                self.max_iter,
                self.tol,
            )
        else:
            raise ValueError("unknow ties")

        self.coef_ = coefs
        self.n_log_likelihood = loss

        if strata is not None:
            self._uses_strata = True
            self.seen_strata = seen_strata

            self._breslow_base_hazard = np.zeros((n_strata, self._max_time_observed))

            for s_i in range(n_strata):
                self._breslow_base_hazard[s_i] = _get_breslow_base_hazard(
                    X_strata[s_i],
                    times_strata[s_i],
                    events_strata[s_i],
                    self._max_time_observed,
                    self.coef_,
                )

            self._breslow_base_survival = np.exp(
                -self._breslow_base_hazard.cumsum(axis=1)
            )
        else:
            self._uses_strata = False

            self._breslow_base_hazard = _get_breslow_base_hazard(
                X, times, events, self._max_time_observed, self.coef_
            )
            self._breslow_base_survival = np.exp(-self._breslow_base_hazard.cumsum())

        self.is_fitted_ = True
        return self

    def predict(self, X, strata=None, max_time: Optional[int] = None):
        check_is_fitted(self)

        if strata is not None:
            strata = _as_int_np_array(strata)

        if max_time is None:
            max_time = self._max_time_observed
        else:
            max_time = _as_int(max_time, "max_time")

        if self._uses_strata:
            if strata is None:
                raise ValueError(
                    "strata must be present if model is trained with strata"
                )

        risk = np.exp(np.dot(X, self.coef_))

        if self._uses_strata:
            strata, has_unseen_strata = map_new_strata(strata, self.seen_strata)

            if has_unseen_strata:
                raise ValueError("predict data has unseen strata")

            survival = self._breslow_base_survival[strata] ** risk[:, None]
            # to do, deal with case where stata key in predict was not present in train
        else:
            survival = self._breslow_base_survival ** risk[:, None]

        if max_time == self._max_time_observed:
            return survival
        elif max_time < self._max_time_observed:
            return survival[:, :max_time]
        else:  # max_time > self._max_time_observed
            missing_dims = max_time - self._max_time_observed

            impulted_values = np.repeat(
                survival[:, -1][np.newaxis], missing_dims, axis=0
            ).T

            return np.hstack([survival, impulted_values])

    def predict_risk(self, X):
        check_is_fitted(self)

        return np.exp(np.dot(X, self.coef_))


class ParametricDiscreteTimePH(_SurvivalPredictBase):
    _parameter_constraints: dict = {
        "distribution": [
            StrOptions(
                {
                    "chen",
                    "weibull",
                    "log_normal",
                    "log_logistic",
                    "gompertz",
                    "additive_chen_weibull",
                }
            )
        ],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "pytensor_mode": [StrOptions({"JAX", "NUMBA"})],
        "strata_uses_pytensor_scan": ["boolean"],
        "coef_prior_normal_sigma": [Interval(Real, 0, None, closed="left")],
        "base_harard_prior_exponential_lam": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        *,
        distribution: Optional[
            Literal[
                "chen",
                "weibull",
                "log_normal",
                "log_logistic",
                "gompertz",
                "additive_chen_weibull",
            ]
        ] = "chen",
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        pytensor_mode: Literal["JAX", "NUMBA"] = "NUMBA",
        strata_uses_pytensor_scan: bool = False,
        coef_prior_normal_sigma: float = 1.5,
        base_harard_prior_exponential_lam: float = 5.0,
    ):
        self.distribution = distribution
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.pytensor_mode = pytensor_mode
        self.strata_uses_pytensor_scan = strata_uses_pytensor_scan
        self.coef_prior_normal_sigma = coef_prior_normal_sigma
        self.base_harard_prior_exponential_lam = base_harard_prior_exponential_lam

    def _get_distribution_function_and_n_prams(self):
        if self.distribution == "chen":
            return _chen_pdf, 2
        elif self.distribution == "weibull":
            return _weibull_pdf, 2
        elif self.distribution == "log_normal":
            return _log_normal_pdf, 2
        elif self.distribution == "log_logistic":
            return _log_logistic_pdf, 2
        elif self.distribution == "gompertz":
            return _gompertz_pdf, 2
        elif self.distribution == "additive_chen_weibull":
            return _additive_chen_weibull_pdf, 4
        else:
            raise ValueError(f"{self.distribution} distribution is not yet implemented")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, times, events, strata=None, check_input=True):

        if check_input:
            X, times, events = validate_survival_data(X, times, events)
            if strata is not None:
                strata = _as_int_np_array(strata)

        self._max_time_observed = np.max(times)

        base_hazard_pdf_callable, n_base_hazard_prams = (
            self._get_distribution_function_and_n_prams()
        )

        if strata is not None:
            self._uses_strata = True
            self.seen_strata = np.unique(strata)

            n_strata = len(self.seen_strata)

            strata, _ = map_new_strata(strata, self.seen_strata)

            coefs, base_hazard_prams = train_parametric_discrete_time_ph_model(
                X,
                times,
                events,
                base_hazard_pdf_callable,
                n_base_hazard_prams,
                self.alpha,
                self.l1_ratio,
                self.pytensor_mode,
                strata,
                n_strata,
                self.strata_uses_pytensor_scan,
                self.coef_prior_normal_sigma,
                self.base_harard_prior_exponential_lam,
            )

        else:
            self._uses_strata = False
            coefs, base_hazard_prams = train_parametric_discrete_time_ph_model(
                X,
                times,
                events,
                base_hazard_pdf_callable,
                n_base_hazard_prams,
                self.alpha,
                self.l1_ratio,
                self.pytensor_mode,
                coef_prior_normal_sigma=self.coef_prior_normal_sigma,
                base_harard_prior_exponential_lam=self.base_harard_prior_exponential_lam,
            )

        self.coef_ = coefs
        self.base_hazard_prams_ = base_hazard_prams

        if all(self.coef_ == 0.0):
            warnings.warn(
                "The model did not train, consider normalizing your features or changing prior parameters"
            )

        self.is_fitted_ = True
        return self

    def predict(self, X, strata=None, max_time: Optional[int] = None):
        check_is_fitted(self)

        if strata is not None:
            strata = _as_int_np_array(strata)

        if max_time is None:
            max_time = self._max_time_observed
        else:
            max_time = _as_int(max_time, "max_time")

        if self._uses_strata:
            if strata is None:
                raise ValueError(
                    "strata must be present if model is trained with strata"
                )
            strata, has_unseen_strata = map_new_strata(strata, self.seen_strata)

            if has_unseen_strata:
                raise ValueError("predict data has unseen strata")

        base_hazard_pdf_callable, _ = self._get_distribution_function_and_n_prams()

        return predict_parametric_discrete_time_ph_model(
            X,
            self.coef_,
            self.base_hazard_prams_,
            max_time,
            base_hazard_pdf_callable,
            strata,
        )

    def predict_risk(self, X):
        check_is_fitted(self)

        return np.exp(np.dot(X, self.coef_))

    def get_base_hazard(self, max_time: Optional[int] = None):

        if max_time is None:
            max_time = self._max_time_observed
        else:
            max_time = _as_int(max_time, "max_time")

        times_of_intrest = np.arange(1, max_time + 1)
        times_of_intrest_norm = _scale_times(times_of_intrest, max_time)

        base_hazard_pdf_callable, _ = self._get_distribution_function_and_n_prams()

        if self._uses_strata:
            return np.apply_along_axis(
                lambda a: base_hazard_pdf_callable(times_of_intrest_norm, a),
                axis=1,
                arr=self.base_hazard_prams_,
            )

        else:
            return base_hazard_pdf_callable(
                times_of_intrest_norm, self.base_hazard_prams_
            )

    def get_pymc_model(
        self,
        X,
        times,
        events,
        max_time: Optional[int] = None,
        labes_names: list[str] | np.ndarray[tuple[int], np.dtype[Any]] | None = None,
        strata: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
        strata_names: list[str] | np.ndarray[tuple[int], np.dtype[Any]] | None = None,
    ):
        base_hazard_pdf_callable, n_base_hazard_prams = (
            self._get_distribution_function_and_n_prams()
        )

        if max_time is None:
            if self.is_fitted_:
                max_time = self._max_time_observed
            else:
                max_time = int(np.max(times))
        else:
            max_time = _as_int(max_time, "max_time")

        if strata is not None:
            seen_strata = np.unique(strata)
            n_strata = len(seen_strata)
            strata, _ = map_new_strata(strata, seen_strata)

        return get_parametric_discrete_time_ph_model(
            X,
            times,
            events,
            base_hazard_pdf_callable,
            n_base_hazard_prams,
            max_time,
            labes_names,
            self.alpha,
            self.l1_ratio,
            strata,
            n_strata,
            strata_names,
            self.strata_uses_pytensor_scan,
            coef_prior_normal_sigma=self.coef_prior_normal_sigma,
            base_harard_prior_exponential_lam=self.base_harard_prior_exponential_lam,
        )


class KaplanMeierSurvivalEstimator(_SurvivalPredictBase):

    def fit(self, X, times, events, strata=None, check_input=True):

        if check_input:
            X, times, events = validate_survival_data(X, times, events)

            if strata is not None:
                strata = _as_int_np_array(strata)

        self._max_time_observed = int(np.max(times))

        if strata is None:

            self._uses_strata = False
            self.kaplan_meier_survival_curve = (
                get_kaplan_meier_survival_curve_from_time_as_int_(
                    events, times, self._max_time_observed
                )
            )
        else:
            (n_strata, seen_strata, events_strata, times_strata, _, _, _, _) = (
                split_and_preprocess_data_by_strata(
                    np.ones((X.shape[0], 1)), times, events, strata
                )
            )

            self._uses_strata = True

            self.seen_strata = seen_strata

            self.kaplan_meier_survival_curve = np.zeros(
                (n_strata, self._max_time_observed)
            )

            for s_i in range(n_strata):
                self.kaplan_meier_survival_curve[s_i] = (
                    get_kaplan_meier_survival_curve_from_time_as_int_(
                        events_strata[s_i],
                        times_strata[s_i],
                        self._max_time_observed,
                    )
                )

        self.is_fitted_ = True
        return self

    def predict(self, X, strata=None, max_time: Optional[int] = None):
        check_is_fitted(self)

        if strata is not None:
            strata = _as_int_np_array(strata)

        if max_time is None:
            max_time = self._max_time_observed
        else:
            max_time = _as_int(max_time, "max_time")

        if self._uses_strata:
            if strata is None:
                raise ValueError(
                    "strata must be present if model is trained with strata"
                )

        kaplan_meier_survival_curve = self.kaplan_meier_survival_curve.copy()

        if max_time < self._max_time_observed:
            if self._uses_strata:
                kaplan_meier_survival_curve = kaplan_meier_survival_curve[:, :max_time]
            else:
                kaplan_meier_survival_curve = kaplan_meier_survival_curve[:max_time]

        elif max_time > self._max_time_observed:
            missing_dims = max_time - self._max_time_observed

            if self._uses_strata:
                impulted_values = np.repeat(
                    kaplan_meier_survival_curve[:, -1][np.newaxis], missing_dims, axis=0
                ).T

                kaplan_meier_survival_curve = np.hstack(
                    [kaplan_meier_survival_curve, impulted_values]
                )

            else:
                impulted_values = np.repeat(
                    kaplan_meier_survival_curve[-1], missing_dims
                )

                kaplan_meier_survival_curve = np.hstack(
                    [kaplan_meier_survival_curve, impulted_values]
                )

        if self._uses_strata:
            strata, has_unseen_strata = map_new_strata(strata, self.seen_strata)

            if has_unseen_strata:
                raise ValueError("predict data has unseen strata")

            return kaplan_meier_survival_curve[strata]
        else:
            return np.repeat(kaplan_meier_survival_curve[None, :], X.shape[0], axis=0)


class KNeighborsSurvival(_SurvivalPredictBase):
    _parameter_constraints: dict = {
        "n_neighbors": [Interval(Integral, 1, None, closed="left"), None],
        "algorithm": [StrOptions({"auto", "ball_tree", "kd_tree", "brute"})],
        "leaf_size": [Interval(Integral, 1, None, closed="left")],
        "p": [Interval(Real, 0, None, closed="right"), None],
        "metric": [
            StrOptions(set(itertools.chain(*VALID_METRICS_KNN.values()))),
            callable,
        ],
        "metric_params": [dict, None],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        n_neighbors: Optional[int] = 10,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: int | float = 2,
        metric: str | Callable = "minkowski",
        metric_param: Optional[dict] = None,
        n_jobs: Optional[int] = None,
    ):

        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_param = metric_param
        self.n_jobs = n_jobs

    def fit(self, X, times, events, check_input=True):
        if check_input:
            X, times, events = validate_survival_data(X, times, events)

        self._max_time_observed = int(np.max(times))

        self._nearestneighbors = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_param,
            n_jobs=self.n_jobs,
        )

        self._nearestneighbors.fit(X)

        self._times_in_memmory = times
        self._events_in_memmory = events

        self.is_fitted_ = True

        return self

    def predict(self, X, max_time: Optional[int] = None):
        check_is_fitted(self)

        if max_time is None:
            max_time = self._max_time_observed
        else:
            max_time = _as_int(max_time, "max_time")

        neighbors_indexes = self._nearestneighbors.kneighbors(X, return_distance=False)

        return build_kaplan_meier_survival_curve_from_neighbors_indexes(
            self._times_in_memmory, self._events_in_memmory, neighbors_indexes, max_time
        )
