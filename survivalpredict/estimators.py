from numbers import Integral, Real
from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ._base_hazard import _get_breslow_base_hazard
from ._cox_ph_estimation import train_cox_ph_breslow, train_cox_ph_efron
from .utils import validate_survival_data


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
    def fit(self, X, times, events, weights=None, check_input=True):

        if check_input:
            X, times, events = validate_survival_data(X, times, events)

        self._max_time_observed = np.max(times)

        if hasattr(self, "coef_"):
            coefs = self.coef_
        else:
            coefs = np.zeros(X.shape[1])

        if self.ties == "breslow":
            coefs, loss = train_cox_ph_breslow(
                X,
                times,
                events,
                self.alpha,
                self.l1_ratio,
                coefs,
                self.max_iter,
                self.tol,
            )
        elif self.ties == "efron":
            coefs, loss = train_cox_ph_efron(
                X,
                times,
                events,
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

        self._breslow_base_hazard = _get_breslow_base_hazard(
            X, times, events, self._max_time_observed, self.coef_
        )
        self._breslow_base_survival = np.exp(-self._breslow_base_hazard.cumsum())

        self.is_fitted_ = True
        return self

    def predict(self, X, max_time: Optional[int] = None):
        check_is_fitted(self)

        if max_time is None:
            max_time = self._max_time_observed
        elif type(max_time) != int:
            raise ValueError("max_time must be an integer")

        risk = np.exp(np.dot(X, self.coef_))

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
