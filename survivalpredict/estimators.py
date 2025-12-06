from numbers import Integral, Real
from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ._base_hazard import _get_breslow_base_hazard
from ._cox_ph_estimation import train_cox_ph_breslow, train_cox_ph_efron
from ._discrete_time_ph_estimation import (
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
from .utils import validate_survival_data, _as_int_np_array
from ._stratification import (
    preprocess_data_for_cox_ph,
    get_l_div_m_stata_per_strata,
    map_new_strata,
)


class SurvivalPredictBase(BaseEstimator):
    pass


class CoxProportionalHazard(SurvivalPredictBase):

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
        elif type(max_time) != int:
            raise ValueError("max_time must be an integer")
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


class ParametricDiscreteTimePH(SurvivalPredictBase):
    _parameter_constraints: dict = {
        "distribution": [
            StrOptions({"chen", "weibull", "log_normal", "log_logistic", "gompertz"})
        ],
    }

    def __init__(
        self,
        *,
        distribution: Optional[
            Literal["chen", "weibull", "log_normal", "log_logistic", "gompertz"]
        ] = "chen",
    ):
        self.distribution = distribution

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
        else:
            raise ValueError(f"{self.distribution} distribution is not yet implemented")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, times, events, weights=None, check_input=True):

        if check_input:
            X, times, events = validate_survival_data(X, times, events)

        self._max_time_observed = np.max(times)

        base_hazard_pdf_callable, n_base_hazard_prams = (
            self._get_distribution_function_and_n_prams()
        )

        coefs, base_hazard_prams = train_parametric_discrete_time_ph_model(
            X, times, events, base_hazard_pdf_callable, n_base_hazard_prams
        )

        self.coef_ = coefs
        self.base_hazard_prams_ = base_hazard_prams

        self.is_fitted_ = True
        return self

    def predict(self, X, max_time: Optional[int] = None):
        check_is_fitted(self)

        if max_time is None:
            max_time = self._max_time_observed
        elif type(max_time) != int:
            raise ValueError("max_time must be an integer")

        base_hazard_pdf_callable, _ = self._get_distribution_function_and_n_prams()

        return predict_parametric_discrete_time_ph_model(
            X,
            self.coef_,
            self.base_hazard_prams_,
            max_time,
            base_hazard_pdf_callable,
        )

    def predict_risk(self, X):
        check_is_fitted(self)

        return np.exp(np.dot(X, self.coef_))

    def get_base_hazard(self, max_time: Optional[int] = None):

        if max_time is None:
            max_time = self._max_time_observed

        times_of_intrest = np.arange(1, max_time + 1)
        times_of_intrest_norm = _scale_times(times_of_intrest, max_time)

        base_hazard_pdf_callable, _ = self._get_distribution_function_and_n_prams()
        return base_hazard_pdf_callable(times_of_intrest_norm, self.base_hazard_prams_)

    def get_pymc_model(self, X, times, events):

        base_hazard_pdf_callable, n_base_hazard_prams = (
            self._get_distribution_function_and_n_prams()
        )

        max_time = times.max()

        return get_parametric_discrete_time_ph_model(
            X, times, events, base_hazard_pdf_callable, max_time, n_base_hazard_prams
        )
