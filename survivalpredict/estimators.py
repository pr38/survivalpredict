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

from ._allen_additive import (
    _estimate_allen_additive_hazard_time_weights,
    _generate_hazards_at_times_from_allen_additive_hazard_weights,
)
from ._base_hazard import _get_breslow_base_hazard
from ._cox_net_ph import get_relative_risk_from_cox_net_ph_weights, train_cox_net_ph
from ._cox_ph_elastic_net import (
    train_cox_elastic_net_regularization_paths,
    train_cox_elastic_net_with_left_censorship,
)
from ._cox_ph_estimation import train_cox_ph_breslow, train_cox_ph_efron
from ._cox_ph_estimation_left_censorship import (
    train_cox_ph_breslow_left_censorship,
    train_cox_ph_breslow_with_left_censorship_scipy_minimize,
    train_cox_ph_efron_left_censorship,
)
from ._data_validation import (
    _as_int,
    _as_int_np_array,
    _as_numeric_np_array,
    validate_survival_data,
    validate_times_start_array,
)
from ._discrete_time_ph_estimation import (
    _additive_chen_weibull_pdf,
    _chen_pdf,
    _gamma_pdf,
    _gompertz_pdf,
    _log_logistic_pdf,
    _log_normal_pdf,
    _scale_times,
    _weibull_pdf,
    get_parametric_discrete_time_ph_model,
    predict_parametric_discrete_time_ph_model,
    train_parametric_discrete_time_ph_model,
)
from ._neighbors import (
    build_kaplan_meier_survival_curve_from_neighbors_indexes,
    build_kaplan_meier_survival_curve_from_neighbors_indexes_with_left_censoring,
)
from ._nonparametric import (
    get_kaplan_meier_survival_curve,
    get_kaplan_meier_survival_curve_with_left_censorship,
)
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
    "CoxNNetPH",
    "AalenAdditiveHazard",
    "CoxElasticNetPH",
]


class _SurvivalPredictBase(BaseEstimator):
    def fit_predict(self, *args, **kwargs):
        """Fit model and Build survival curves."""
        predict_kwargs = kwargs.copy()

        if "max_time" in kwargs:
            del kwargs["max_time"]

        if "times_start" in predict_kwargs:
            del predict_kwargs["times_start"]

        X = args[0]

        return self.fit(*args, **kwargs).predict(X, **predict_kwargs)


class CoxProportionalHazard(_SurvivalPredictBase):
    """
    Cox Proportional Hazards.

    The ‘Cox Proportional Hazards’ model is a linear semi-parametric relative
    risk model. A staple of survival analysis. Cox more or less trains on
    ranking relative risk to estimate its coefficients. After training, the
    'Breslow estimator' is run on relative risk and events over time to build
    the base hazard. A product of the relative risk and base hazard at each
    point in time is used to build the survival curves.

    The Cox is called ‘semi-parametric’ due to the fact that is does not
    directly estimate the hazard, but only relative hazard. Hence,
    ‘partial-likelihood’ is what Cox estimates maximizes.

    Parameters
    ----------
    alpha : float, default=0.0
        Constant that multiplies the penalty terms. Used to penalize
        coefficients durring training.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    max_iter : Optional[int], default=100
        The maximum number of iterations.

    ties : {"breslow", "efron"}, default='breslow'
        The method to handle ‘tied’ event times. Cox’s coefficients are
        intended to represent the relative risk of observations in proportion
        to each other, independent of time. The presence of ‘tied’ or
        concurrent failures muddies the interpretability of Cox’s
        coefficients.‘Breslow ties’ ignore said issue and perform best on
        predictions. ‘Efron ties’ shaves some of the influence of some tied
        data on the likelihood in hopes of solving said problem, at the price
        of prediction performance. Use Breslow if prediction performance is
        your primary concern, and use Efron in cases of inference.

    tol : float, default=1e-9
        The tolerance for the optimization: if the updates are smaller or equal
        to ``tol``, the optimization code checks the dual gap for optimality
        and continues until it is smaller or equal to ``tol``.

    Attributes
    ----------
    coef_ : ndarray of ndarray of shape (n_features,)
        Coefficients of the model.

    n_log_likelihood : float
        Negative log likelihood of the model at the point of convergence.

    _breslow_base_hazard : ndarray of ndarray of shape (max_time_seen,) or shape (n_strata,max_time_seen)
        Base hazard generated from training data, used for predicting survival
        curves.

    _breslow_base_survival : ndarray of ndarray of shape (max_time_seen,) or shape (n_strata,max_time_seen)
        Base survival generated from training data, used for predicting
        survival curves.
    """

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
    def fit(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        times: np.ndarray[tuple[int], np.dtype[np.int64]],
        events: np.ndarray[tuple[int], np.dtype[np.bool_]],
        strata: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
        check_input: bool = True,
        times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
    ):
        """
        Fit model.

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

        check_input : bool, default=True
            If True, validates and casts inputs.

        times_start : array-like of shape (n_samples, dtype=np.int64), default=None
            Starting point for observation. If not passed in, all times_start
            times are assumed to be 0.

        Returns
        -------
        object
            Fitted Estimator.
        """

        use_left_censorship = times_start is not None

        if check_input:
            X, times, events = validate_survival_data(X, times, events)

            if strata is not None:
                strata = _as_int_np_array(strata, "strata")

            if use_left_censorship:
                times_start = validate_times_start_array(times_start, times)

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
            if use_left_censorship:
                times_start = times_start[argsort]

        if use_left_censorship:
            (
                n_strata,
                seen_strata,
                X_strata,
                times_strata,
                events_strata,
                time_return_inverse_strata,
                n_unique_times_strata,
                event_counts_at_times_strata,
                _,
                time_start_return_inverse_strata,
            ) = preprocess_data_for_cox_ph(X, times, events, strata, times_start)

            l_div_m_stata = get_l_div_m_stata_per_strata(
                events_strata,
                times_strata,
                time_return_inverse_strata,
                n_unique_times_strata,
            )

            if self.ties == "efron":
                coefs, loss = train_cox_ph_efron_left_censorship(
                    n_strata,
                    X_strata,
                    events_strata,
                    n_unique_times_strata,
                    l_div_m_stata,
                    time_return_inverse_strata,
                    time_start_return_inverse_strata,
                    self.alpha,
                    self.l1_ratio,
                    coefs,
                    self.max_iter,
                    self.tol,
                )
            elif self.ties == "breslow":

                coefs, loss = train_cox_ph_breslow_left_censorship(
                    X_strata,
                    events_strata,
                    n_unique_times_strata,
                    event_counts_at_times_strata,
                    time_return_inverse_strata,
                    time_start_return_inverse_strata,
                    n_strata,
                    self.alpha,
                    self.l1_ratio,
                    coefs,
                    self.max_iter,
                    self.tol,
                )

            else:
                raise ValueError("unknow ties")

        else:  # no left censorship
            (
                n_strata,
                seen_strata,
                X_strata,
                times_strata,
                events_strata,
                time_return_inverse_strata,
                n_unique_times_strata,
                event_counts_at_times_strata,
                _,
                _,
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
                risk = np.exp(np.dot(X_strata[s_i], self.coef_))
                self._breslow_base_hazard[s_i] = _get_breslow_base_hazard(
                    risk,
                    times_strata[s_i],
                    events_strata[s_i],
                    self._max_time_observed,
                )

            self._breslow_base_survival = np.exp(
                -self._breslow_base_hazard.cumsum(axis=1)
            )
        else:
            self._uses_strata = False

            risk = np.exp(np.dot(X_strata[0], self.coef_))
            self._breslow_base_hazard = _get_breslow_base_hazard(
                risk, times, events, self._max_time_observed
            )
            self._breslow_base_survival = np.exp(-self._breslow_base_hazard.cumsum())

        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        strata=None,
        max_time: Optional[int] = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        Build survival curves on an array of vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predicting data.

        strata : array-like of shape (n_samples,), dtype=np.int64, default=None
            If passed in, associated strata for per observation.

        max_time : int, default=None
            Maximum time of built survival curves. If none, maximum time is max
            time seen on training data.

        Returns
        -------
        ndarray of shape (n_samples, max_time), dtype=np.float64
            The estimated survival curves, the left-most column is the probability of survival at time 1,
            and the right-most column ends at max_time.
        """

        check_is_fitted(self)

        if strata is not None:
            strata = _as_int_np_array(strata, "strata")

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

    def predict_risk(
        self, X: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Build relative risk on an array of vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predicting data.

        Returns
        -------
        ndarray of shape (n_samples), dtype=np.float64
            The Relative risk of X, used under the hood for building survival curves.
            Relative risk is what 'Concordance Index' examines.
        """

        check_is_fitted(self)

        return np.exp(np.dot(X, self.coef_))


class ParametricDiscreteTimePH(_SurvivalPredictBase):
    """
    Parametric Discrete Time Proportional Hazards.

    A fully parametric linear proportional hazards model. Unlike Cox, both the
    coefficients and the base hazard are directly estimated from observed
    survival over time. Various distributions are available as base hazards;
    namely, Chen, Weibull, Log-Normal, Log-logistic, Gompertz, Gamma and
    Additive-Chen-Weibull[1] are available as hyperparameters. Maximum
    likelihood is estimated using a survival distinct time likelihood[2] with
    censorship. Implemented with Pymc/Pytensor, with either a Jax or numba
    backend.

    Parameters
    ----------
    distribution : {"chen", "weibull","log_normal","log_logistic","gamma","gompertz","additive_chen_weibull"}, default='chen'
        Distribution of base hazard.

    alpha : float, default=0.0
        Constant that multiplies the penalty terms. Used to penalize
        coefficients durring training.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    pytensor_mode : {"JAX", "NUMBA","FAST_COMPILE"}, default='JAX'
        Pytensor backend. ‘JAX’ has the fastest compile time, but is not multiprocessing safe.
        NUMBA is multiprocessing safe, but has a long compile time. Pytensor's ‘FAST_COMPILE’
        mode is multiprocessing safe and has a fast compile time, but runs slower than the other modes.
        ‘JAX’ is a good default, but ‘NUMBA’ is recommended when using multiprocessing.

    strata_uses_pytensor_scan : bool, default=False
        If strata are present and ‘strata_uses_pytensor_scan’ is True,
        Pytensor's 'scan' functionality is used to map strata to observations
        during training. Using Pytensor scan might increase the Pytensor
        compile time, but will lead to a faster runtime. For considerable data
        and a high quantity of starta, it is recommended to set
        strata_uses_pytensor_scan to True.

    coef_prior_normal_sigma : float, default=1.5
        This class runs a Pymc model under the hood. The coefficients are
        modeled as normal distributions. This parameter is the sigma of the
        prior. The larger the sigma, the wider the possible set of values
        coverage. It is recommended to scale the data to avoid tuning this
        parameter.

    base_harard_prior_exponential_lam : float, default=5.0
        This class runs a Pymc model under the hood. The base hazard
        distrabution's parameters are modeled as exponential distributions.
        This parameter is the 'lam' of the prior of the base hazard
        distrabution' parameters. It is recommended to scale the data to avoid
        tuning this parameter.

    scipy_minimize_method : {"nelder-mead","powell","CG","BFGS","Newton-CG","L-BFGS-B","TNC","COBYLA","SLSQP","trust-constr","dogleg","trust-ncg","trust-exact","trust-krylov","basinhopping",}, default='L-BFGS-B'
        This class runs a Pymc model under the hood. Durring training we simply find the 'Maximum likelihood estimation'(MLE)/
        'Maximum a posteriori estimation'(MAP). This is the exposed method that PYMC find the MLE/MAP.

    References
    ----------

    [1] Thanh Thach T, Briš R. An additive Chen-Weibull distribution and its
    applications in reliability modeling. Qual Reliab Engng Int.
    2021;37:352–373. https://doi.org/10.1002/qre.2740

    [2] Suresh K, Severn C, Ghosh D. Survival prediction models: an
    introduction to discrete-time modeling. BMC Med Res Methodol. 2022 Jul
    26;22(1):207. doi: 10.1186/s12874-022-01679-6. PMID: 35883032; PMCID:
    PMC9316420.
    """

    _parameter_constraints: dict = {
        "distribution": [
            StrOptions(
                {
                    "chen",
                    "weibull",
                    "log_normal",
                    "log_logistic",
                    "gamma",
                    "gompertz",
                    "additive_chen_weibull",
                }
            )
        ],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "pytensor_mode": [StrOptions({"JAX", "NUMBA", "FAST_COMPILE"})],
        "strata_uses_pytensor_scan": ["boolean"],
        "coef_prior_normal_sigma": [Interval(Real, 0, None, closed="left")],
        "base_harard_prior_exponential_lam": [Interval(Real, 0, None, closed="left")],
        "scipy_minimize_method": [
            StrOptions(
                {
                    "nelder-mead",
                    "powell",
                    "CG",
                    "BFGS",
                    "Newton-CG",
                    "L-BFGS-B",
                    "TNC",
                    "COBYLA",
                    "SLSQP",
                    "trust-constr",
                    "dogleg",
                    "trust-ncg",
                    "trust-exact",
                    "trust-krylov",
                    "basinhopping",
                }
            )
        ],
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
                "gamma",
                "gompertz",
                "additive_chen_weibull",
            ]
        ] = "chen",
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        pytensor_mode: Literal["JAX", "NUMBA", "FAST_COMPILE"] = "JAX",
        strata_uses_pytensor_scan: bool = False,
        coef_prior_normal_sigma: float = 1.5,
        base_harard_prior_exponential_lam: float = 5.0,
        scipy_minimize_method: Literal[
            "nelder-mead",
            "powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
            "basinhopping",
        ] = "L-BFGS-B",
    ):
        self.distribution = distribution
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.pytensor_mode = pytensor_mode
        self.strata_uses_pytensor_scan = strata_uses_pytensor_scan
        self.coef_prior_normal_sigma = coef_prior_normal_sigma
        self.base_harard_prior_exponential_lam = base_harard_prior_exponential_lam
        self.scipy_minimize_method = scipy_minimize_method

    def _get_distribution_function_and_n_prams(self) -> tuple[Callable, int]:
        if self.distribution == "chen":
            return _chen_pdf, 2
        elif self.distribution == "weibull":
            return _weibull_pdf, 2
        elif self.distribution == "log_normal":
            return _log_normal_pdf, 2
        elif self.distribution == "log_logistic":
            return _log_logistic_pdf, 2
        elif self.distribution == "gamma":
            return _gamma_pdf, 2
        elif self.distribution == "gompertz":
            return _gompertz_pdf, 2
        elif self.distribution == "additive_chen_weibull":
            return _additive_chen_weibull_pdf, 4
        else:
            raise ValueError(f"{self.distribution} distribution is not yet implemented")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        times: np.ndarray[tuple[int], np.dtype[np.int64]],
        events: np.ndarray[tuple[int], np.dtype[np.bool_]],
        strata: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
        check_input: bool = True,
        times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
    ):
        """
        Fit model.

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

        check_input : bool, default=True
            If True, validates and casts inputs.

        times_start : array-like of shape (n_samples, dtype=np.int64), default=None
            Starting point for observation. If not passed in, all times_start times are assumed to be 0.

        Returns
        -------
        object
            Fitted Estimator.
        """

        if check_input:
            X, times, events = validate_survival_data(X, times, events)
            if strata is not None:
                strata = _as_int_np_array(strata, "strata")
            if times_start is not None:
                times_start = validate_times_start_array(times_start, times)

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
                self.scipy_minimize_method,
                times_start=times_start,
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
                times_start=times_start,
            )

        self.coef_ = coefs
        self.base_hazard_prams_ = base_hazard_prams

        if all(self.coef_ == 0.0):
            warnings.warn(
                "The model did not train, consider normalizing your features or changing prior parameters"
            )

        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        strata=None,
        max_time: Optional[int] = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        Build survival curves on an array of vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predicting data.

        strata : array-like of shape (n_samples,), dtype=np.int64, default=None
            If passed in, associated strata for per observation.

        max_time : Optional[int], default=None
            Maximum time of built survival curves. If none, maximum time is max
            time seen on training data.

        Returns
        -------
        ndarray of shape (n_samples, max_time), dtype=np.float64
            The estimated survival curves, the left-most column is the
            probability of survival at time 1, and the right-most column ends
            at max_time.
        """

        check_is_fitted(self)

        if strata is not None:
            strata = _as_int_np_array(strata, "strata")

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
            self._max_time_observed,
            max_time,
            base_hazard_pdf_callable,
            strata,
        )

    def predict_risk(
        self, X: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Build relative risk on an array of vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predicting data.

        Returns
        -------
        ndarray of shape (n_samples), dtype=np.float64
            The Relative risk of X, used under the hood for building survival curves.
            Relative risk is what 'Concordance Index' examines.
        """

        check_is_fitted(self)

        return np.exp(np.dot(X, self.coef_))

    def get_base_hazard(
        self, max_time: Optional[int] = None
    ) -> (
        np.ndarray[tuple[int, int], np.dtype[np.float64]]
        | np.ndarray[tuple[int], np.dtype[np.float64]]
    ):
        """
        Retrieve base hazards estimated by model.

        Parameters
        ----------
        max_time : int, default=None
            Maximum time of built survival curves. If none, maximum time is max time seen on training data.

        Returns
        -------
        ndarray of shape (max_time,) or (n_strata,max_time,)  , dtype=np.float64
            The estimated base hazard; used in building survival curves.
        """

        check_is_fitted(self)

        if max_time is None:
            max_time = self._max_time_observed
        else:
            max_time = _as_int(max_time, "max_time")

        times_of_intrest = np.arange(1, max_time + 1)
        times_of_intrest_norm = _scale_times(times_of_intrest, self._max_time_observed)

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
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        times: np.ndarray[tuple[int], np.dtype[np.int64]],
        events: np.ndarray[tuple[int], np.dtype[np.bool_]],
        max_time: Optional[int] = None,
        labes_names: list[str] | np.ndarray[tuple[int], np.dtype[Any]] | None = None,
        strata: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
        strata_names: list[str] | np.ndarray[tuple[int], np.dtype[Any]] | None = None,
        times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
    ) -> "pymc.Model":
        """
        Return the underlying Pymc model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        times : array-like of shape (n_samples), dtype=np.int64
            Point in time last observed.

        events : array-like of shape (n_samples), dtype=np.bool_
            Experianed event.

        max_time : int, default=None
            Maximum time of built survival curves. If none, maximum time is max
            time seen on training data.

        labes_names : list of str, default=None
            Names for feature, allows for parameters associated with each
            feature to be named accordingly.

        strata : array-like of shape (n_samples,), dtype=np.int64, default=None
            If passed in, associated strata for per observation.

        strata_names : list of str, default=None
            Names for strata, allows for parameters associated with each strata to be named accordingly.

        times_start : array-like of shape (n_samples, dtype=np.int64), default=None
            Starting point for observation. If not passed in, all times_start times are assumed to be 0.

        Returns
        -------
        "pymc.Model"
        """

        base_hazard_pdf_callable, n_base_hazard_prams = (
            self._get_distribution_function_and_n_prams()
        )

        X, times, events = validate_survival_data(X, times, events)
        if strata is not None:
            strata = _as_int_np_array(strata, "strata")

        if max_time is None:
            if hasattr(self, "is_fitted_"):
                max_time = self._max_time_observed
            else:
                max_time = int(np.max(times))
        else:
            max_time = _as_int(max_time, "max_time")

        if strata is not None:
            seen_strata = np.unique(strata)
            n_strata = len(seen_strata)
            strata, _ = map_new_strata(strata, seen_strata)
        else:
            n_strata = None

        if times_start is not None:
            times_start = validate_times_start_array(times_start, times)

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
            times_start=times_start,
        )


class KaplanMeierSurvivalEstimator(_SurvivalPredictBase):
    """
    The Kaplan-Meier estimate of the survival estimation.

    Kaplan-Meier is a univariate non-parametric survival curve estimation. It
    can be useful as a baseline/dummy estimator.
    """

    def fit(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        times: np.ndarray[tuple[int], np.dtype[np.int64]],
        events: np.ndarray[tuple[int], np.dtype[np.bool_]],
        strata: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
        check_input: bool = True,
        times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
    ):
        """
        Fit model.

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

        check_input : bool, default=True
            If True, validates and casts inputs.

        times_start : array-like of shape (n_samples, dtype=np.int64), default=None
            Starting point for observation. If not passed in, all times_start times are assumed to be 0.

        Returns
        -------
        object
            Fitted Estimator.
        """

        if times_start is None:
            uses_left_censorship = False
        else:
            uses_left_censorship = True

        if check_input:
            X, times, events = validate_survival_data(X, times, events)

            if strata is not None:
                strata = _as_int_np_array(strata, "strata")
            if uses_left_censorship:
                times_start = validate_times_start_array(times_start, times)

        self._max_time_observed = int(np.max(times))

        if not uses_left_censorship:
            times_start = np.zeros(X.shape[0], dtype=np.int64)

        if strata is None:

            self._uses_strata = False
            if uses_left_censorship == False:
                self.kaplan_meier_survival_curve = get_kaplan_meier_survival_curve(
                    events, times, self._max_time_observed
                )
            else:
                self.kaplan_meier_survival_curve = (
                    get_kaplan_meier_survival_curve_with_left_censorship(
                        events, times, times_start, self._max_time_observed
                    )
                )
        else:
            (
                n_strata,
                seen_strata,
                events_strata,
                times_strata,
                _,
                _,
                _,
                _,
                times_start_strata,
                _,
            ) = split_and_preprocess_data_by_strata(
                np.ones((X.shape[0], 1)),
                times,
                events,
                strata,
                times_start,
                uses_left_censorship,
            )

            self._uses_strata = True

            self.seen_strata = seen_strata

            self.kaplan_meier_survival_curve = np.zeros(
                (n_strata, self._max_time_observed)
            )

            for s_i in range(n_strata):
                if uses_left_censorship == False:
                    self.kaplan_meier_survival_curve[s_i] = (
                        get_kaplan_meier_survival_curve(
                            events_strata[s_i],
                            times_strata[s_i],
                            self._max_time_observed,
                        )
                    )
                else:
                    self.kaplan_meier_survival_curve[s_i] = (
                        get_kaplan_meier_survival_curve_with_left_censorship(
                            events_strata[s_i],
                            times_strata[s_i],
                            times_start_strata[s_i],
                            self._max_time_observed,
                        )
                    )

        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        strata: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
        max_time: Optional[int] = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        Build survival curves on an array of vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predicting data.

        strata : array-like of shape (n_samples,), dtype=np.int64, default=None
            If passed in, associated strata for per observation.

        max_time : int, default=None
            Maximum time of built survival curves. If none, maximum time is max time seen on training data.

        Returns
        -------
        ndarray of shape (n_samples, max_time), dtype=np.float64
            The estimated survival curves, the left-most column is the
            probability of survival at time 1, and the right-most column ends
            at max_time.
        """

        check_is_fitted(self)

        if strata is not None:
            strata = _as_int_np_array(strata, "strata")

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
    """
    Survival curves implementing the k-nearest neighbors vote.

    Parameters docs are taken from scikit-learn.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the speed of
        the construction and query, as well as the memory required to store the
        tree.  The optimal value depends on the nature of the problem.

    p : float, default=2
        Power parameter for the Minkowski metric. When p = 1, this is equivalent
        to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
        For arbitrary p, minkowski_distance (l_p) is used. This parameter is expected
        to be positive.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_
        and the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

    metric_param : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.
    """

    _parameter_constraints: dict = {
        "n_neighbors": [Interval(Integral, 1, None, closed="left"), None],
        "algorithm": [StrOptions({"auto", "ball_tree", "kd_tree", "brute"})],
        "leaf_size": [Interval(Integral, 1, None, closed="left")],
        "p": [Interval(Real, 0, None, closed="right"), None],
        "metric": [
            StrOptions(set(itertools.chain(*VALID_METRICS_KNN.values()))),
            callable,
        ],
        "metric_param": [dict, None],
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

    def fit(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        times: np.ndarray[tuple[int], np.dtype[np.int64]],
        events: np.ndarray[tuple[int], np.dtype[np.bool_]],
        check_input: bool = True,
        times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
    ):
        """
        Fit model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        times : array-like of shape (n_samples), dtype=np.int64
            Point in time last observed.

        events : array-like of shape (n_samples), dtype=np.bool_
            Experianed event.

        check_input : bool, default=True
            If True, validates and casts inputs.

        times_start : array-like of shape (n_samples, dtype=np.int64), default=None
            Starting point for observation. If not passed in, all times_start times are assumed to be 0.

        Returns
        -------
        object
            Fitted Estimator.
        """

        if times_start is not None:
            self._uses_times_start = True
        else:
            self._uses_times_start = False

        if check_input:
            X, times, events = validate_survival_data(X, times, events)
            validate_times_start_array

            if self._uses_times_start:
                times_start = validate_times_start_array(times_start, times)

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

        if self._uses_times_start:
            self._times_start_in_memmory = times_start

        self.is_fitted_ = True

        return self

    def predict(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        max_time: Optional[int] = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        Build survival curves on an array of vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predicting data.

        max_time : int, default=None
            Maximum time of built survival curves. If none, maximum time is max
            time seen on training data.

        Returns
        -------
        ndarray of shape (n_samples, max_time), dtype=np.float64
            The estimated survival curves, the left-most column is the
            probability of survival at time 1, and the right-most column ends
            at max_time.
        """

        check_is_fitted(self)

        if max_time is None:
            max_time = self._max_time_observed
        else:
            max_time = _as_int(max_time, "max_time")

        neighbors_indexes = self._nearestneighbors.kneighbors(X, return_distance=False)

        X = _as_numeric_np_array(X)

        if self._uses_times_start:
            return build_kaplan_meier_survival_curve_from_neighbors_indexes_with_left_censoring(
                self._times_in_memmory,
                self._events_in_memmory,
                neighbors_indexes,
                max_time,
                self._times_start_in_memmory,
            )
        else:
            return build_kaplan_meier_survival_curve_from_neighbors_indexes(
                self._times_in_memmory,
                self._events_in_memmory,
                neighbors_indexes,
                max_time,
            )


class CoxNNetPH(_SurvivalPredictBase):
    """
    Artificial neural network proportional hazards Model.

    A neural network model for estimating relative risk. Cox proportional
    hazards model's 'negative log likelihood for Breslow ties' is used as a
    loss function. Breslow's base hazard for relative risk is used to estimate
    survival across time. The combination of relative risk and base hazard is
    used to generate survival curves, like Cox proportional hazards.
    Implemented using Jax. All activation functions as assumed to be relu.

    Parameters
    ----------
    hidden_layers : list of ints, default=[100,]
        The ith element represents the number of neurons in the ith hidden
        layer.

    alpha : float, default=0.0
        Constant that multiplies the penalty terms. Used to penalize
        coefficients durring training.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    init_dis : Literal["uniform", "normal"], default="uniform"
        Distribution of the He/Kaiming weight initialization.

    track_loss : bool, default=True
        If True, changes to loss over training itteration is tracked.

    max_iter : int, default=100
        The maximum number of iterations.

    gradient_updater : {"adadelta", "adagrad", "adam", "adamax", "rmsprop"}, default="adam"
        Gradient updating strategy, used for training. Corresponds to optax
        Optimizers.

    learning_rate : float, default=0.01
        Corresponds to optax Optimizer parameter.

    beta1 : float, default=0.9
        Corresponds to optax Optimizer parameter.

    beta2 : float, default=0.999
        Corresponds to optax Optimizer parameter.

    epsilon : float, default=0.0000001
        Corresponds to optax Optimizer parameter.

    rho : float, default=0.95
        Corresponds to optax Optimizer parameter.

    decay : float, default=0.9
        Corresponds to optax Optimizer parameter.

    Attributes
    ----------
    coef_ : ndarray of ndarray of shape (n_features,)
        Coefficients of the model.

    _loss : float
        Negative log likelihood of the model at the point of convergence.

    _breslow_base_hazard : ndarray of ndarray of shape (max_time_seen,) or shape (n_strata,max_time_seen)
        Base hazard generated from training data, used for predicting survival
        curves.

    _breslow_base_survival : ndarray of ndarray of shape (max_time_seen,) or shape (n_strata,max_time_seen)
        Base survival generated from training data, used for predicting
        survival curves.

    losses_per_steps : list of float
        If track_loss is set to True, loss at each itteration while training.
    """

    _parameter_constraints: dict = {
        "hidden_layers": [list],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "init_dis": [StrOptions({"uniform", "normal"})],
        "track_loss": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "gradient_updater": [
            StrOptions({"adadelta", "adagrad", "adam", "adamax", "rmsprop"})
        ],
        "learning_rate": [Interval(Real, 0, None, closed="left")],
        "beta1": [Interval(Real, 0, None, closed="left")],
        "beta2": [Interval(Real, 0, None, closed="left")],
        "rho": [Interval(Real, 0, None, closed="left")],
        "decay": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        hidden_layers: list[int] = [100],
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        init_dis: Literal["uniform", "normal"] = "uniform",
        track_loss=True,
        max_iter=100,
        gradient_updater: Literal[
            "adadelta",
            "adagrad",
            "adam",
            "adamax",
            "rmsprop",
        ] = "adam",
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 0.0000001,
        rho: float = 0.95,
        decay: float = 0.9,
    ):
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.init_dis = init_dis
        self.track_loss = track_loss
        self.max_iter = max_iter
        self.gradient_updater = gradient_updater
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.rho = rho
        self.decay = decay

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        times: np.ndarray[tuple[int], np.dtype[np.int64]],
        events: np.ndarray[tuple[int], np.dtype[np.bool_]],
        strata: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
        check_input: bool = True,
        times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
    ):
        """
        Fit model.

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

        check_input : bool, default=True
            If True, validates and casts inputs.

        times_start : array-like of shape (n_samples, dtype=np.int64), default=None
            Starting point for observation. If not passed in, all times_start
            times are assumed to be 0.

        Returns
        -------
        object
            Fitted Estimator.
        """

        uses_left_censorship = times_start is not None

        self._uses_strata = strata is not None

        if check_input:
            X, times, events = validate_survival_data(X, times, events)

            if self._uses_strata:
                strata = _as_int_np_array(strata, "strata")

            if uses_left_censorship:
                times_start = validate_times_start_array(times_start, times)

        for i in self.hidden_layers:
            if type(i) != int:
                raise ValueError("hidden_layers must be a list of ints")

        self._max_time_observed = np.max(times)

        (
            n_strata,
            seen_strata,
            X_strata,
            times_strata,
            events_strata,
            time_return_inverse_strata,
            n_unique_times_strata,
            _,
            _,
            time_start_return_inverse_strata,
        ) = preprocess_data_for_cox_ph(X, times, events, strata, times_start)

        if uses_left_censorship is False:
            time_start_return_inverse_strata = None

        if hasattr(self, "coef_"):
            coef = self.coef_
        else:
            coef = None

        coef, loss, losses_per_steps = train_cox_net_ph(
            X_strata,
            n_strata,
            events_strata,
            time_return_inverse_strata,
            n_unique_times_strata,
            self.hidden_layers,
            coef,
            self.alpha,
            self.l1_ratio,
            self.init_dis,
            self.track_loss,
            self.max_iter,
            self.gradient_updater,
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.rho,
            self.decay,
            time_start_return_inverse_strata,
        )

        self.coef_ = coef
        self._loss = loss
        self._losses_per_steps = losses_per_steps

        if self._uses_strata:
            self.seen_strata = seen_strata

            self._breslow_base_hazard = np.zeros((n_strata, self._max_time_observed))

            for s_i in range(n_strata):
                risk = get_relative_risk_from_cox_net_ph_weights(
                    X_strata[s_i], self.coef_
                )
                self._breslow_base_hazard[s_i] = _get_breslow_base_hazard(
                    risk,
                    times_strata[s_i],
                    events_strata[s_i],
                    self._max_time_observed,
                )

            self._breslow_base_survival = np.exp(
                -self._breslow_base_hazard.cumsum(axis=1)
            )

        else:
            risk = get_relative_risk_from_cox_net_ph_weights(X, self.coef_)
            self._breslow_base_hazard = _get_breslow_base_hazard(
                risk, times, events, self._max_time_observed
            )
            self._breslow_base_survival = np.exp(-self._breslow_base_hazard.cumsum())

        self.is_fitted_ = True

        return self

    def predict(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        strata: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
        max_time: Optional[int] = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        Build survival curves on an array of vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predicting data.

        strata : array-like of shape (n_samples,), dtype=np.int64, default=None
            If passed in, associated strata for per observation.

        max_time : int, default=None
            Maximum time of built survival curves. If none, maximum time is max
            time seen on training data.

        Returns
        -------
        ndarray of shape (n_samples, max_time), dtype=np.float64
            The estimated survival curves, the left-most column is the
            probability of survival at time 1, and the right-most column ends
            at max_time.
        """

        check_is_fitted(self)

        if max_time is None:
            max_time = self._max_time_observed
        else:
            max_time = _as_int(max_time, "max_time")

        X = _as_numeric_np_array(X)

        risk = get_relative_risk_from_cox_net_ph_weights(X, self.coef_)

        if self._uses_strata:
            if strata is None:
                raise ValueError(
                    "strata must be present if model is trained with strata"
                )

            strata, has_unseen_strata = map_new_strata(strata, self.seen_strata)

            if has_unseen_strata:
                raise ValueError("predict data has unseen strata")

            base_survival = self._breslow_base_survival

            if max_time < self._max_time_observed:
                base_survival = base_survival[:, :max_time]
            elif max_time > self._max_time_observed:
                missing_len = max_time - self._max_time_observed

                impulted_values = np.repeat(
                    base_survival[:, -1][np.newaxis], missing_len, axis=0
                ).T

                base_survival = np.hstack([base_survival, impulted_values])

            return base_survival[strata] ** risk[:, None]

        else:

            base_survival = self._breslow_base_survival

            if max_time < self._max_time_observed:
                base_survival = base_survival[:max_time]
            elif max_time > self._max_time_observed:
                missing_len = max_time - self._max_time_observed

                impulted_values = np.repeat(base_survival[-1], missing_len, axis=0)

                base_survival = np.concat([base_survival, impulted_values])

            return base_survival ** risk[:, None]

    def predict_risk(
        self, X: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Build relative risk on an array of vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predicting data.

        Returns
        -------
        ndarray of shape (n_samples), dtype=np.float64
            The Relative risk of X, used under the hood for building survival
            curves. Relative risk is what 'Concordance Index' examines.
        """

        check_is_fitted(self)

        X = np.array(X)

        return get_relative_risk_from_cox_net_ph_weights(X, self.coef_)


class AalenAdditiveHazard(_SurvivalPredictBase):
    """
    Aalen Additive Hazards.

    Aalen Additive Hazards is a linear multivariate non-parametric estimation
    of hazard. It allows for each interval of time and feature to have an
    associated coefficient, allowing for the effects of features to change over
    time. Aalen Additive Hazards simply runs (ridge) linear regressions at
    different points in time, 1 if the event and 0 otherwise , censored times
    are ignored. The results of the linear regressions are used for generating
    hazards and ultimately survival curves. AalenAdditiveHazard models are
    tricky to train.

    Parameters
    ----------
    clip_hazards : bool, default=True
        If True, clips hazards to be between 0 and 1. Ensuring that hazard
        values are realistic.

    alpha : float, default=0.0
        Constant that multiplies the penalty terms. Used to penalize
        coefficients durring training.

    Attributes
    ----------
    _hazard_weights : ndarray of ndarray of shape (n_features,n)
        Coefficients of the model.

    _hazard_weights_times : ndarray of ndarray of shape (n)
        Times associated for each interval of time in the  _hazard_weights
        array.
    """

    _parameter_constraints: dict = {
        "clip_hazards": ["boolean"],
        "alpha": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(self, clip_hazards: bool = True, alpha: float = 0.0):
        self.clip_hazards = clip_hazards
        self.alpha = alpha

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        times: np.ndarray[tuple[int], np.dtype[np.int64]],
        events: np.ndarray[tuple[int], np.dtype[np.int64]],
        check_input: bool = True,
        times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
    ):
        """
        Fit model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        times : array-like of shape (n_samples), dtype=np.int64
            Point in time last observed.

        events : array-like of shape (n_samples), dtype=np.bool_
            Experianed event.

        check_input : bool, default=True
            If True, validates and casts inputs.

        times_start : array-like of shape (n_samples, dtype=np.int64), default=None
            Starting point for observation. If not passed in, all times_start
            times are assumed to be 0.

        Returns
        -------
        object
            Fitted Estimator.
        """

        if check_input:
            X, times, events = validate_survival_data(X, times, events)

        if times_start is None:
            times_start = np.zeros_like(times, dtype=np.int64)
        else:
            times_start = validate_times_start_array(times_start, times)

        self._max_time_observed = int(np.max(times))

        hazard_weights, hazard_weights_times = (
            _estimate_allen_additive_hazard_time_weights(
                X, times, events, times_start, self.alpha
            )
        )

        self._hazard_weights = hazard_weights
        self._hazard_weights_times = hazard_weights_times

        self.is_fitted_ = True

        return self

    def predict(
        self, X, max_time: Optional[int] = None
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Build survival curves on an array of vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predicting data.

        max_time : Optional[int], default=None
            Maximum time of built survival curves. If none, maximum time is max
            time seen on training data.

        Returns
        -------
        ndarray of shape (n_samples, max_time), dtype=np.float64
            The estimated survival curves, the left-most column is the
            probability of survival at time 1, and the right-most column ends
            at max_time.
        """

        check_is_fitted(self)

        X = _as_numeric_np_array(X)

        if max_time is None:
            max_time = self._max_time_observed
        else:
            max_time = _as_int(max_time, "max_time")

        hazards = _generate_hazards_at_times_from_allen_additive_hazard_weights(
            X, self._hazard_weights, self._hazard_weights_times, max_time
        )

        if self.clip_hazards:
            hazards = np.clip(hazards, 0.0, 1.0)

        return np.exp(-hazards.cumsum(axis=1))


class CoxElasticNetPH(_SurvivalPredictBase):
    """
    Cox Proportional Hazards with Elastic Net penalty and feature shrinkage.

    A Cox Proportional Hazards model with Elastic Net penalty estimated via
    coordinate descent. The coordinate descent algorithm for Elastic Net/Lasso
    allows shrinking features asynchronously as the ‘alpha’ parameter
    increases, and the ‘l1_ratio’ is greater than 0. The raphson-newton-like
    for coordinate descent described in Simon et al. (2011)[1] is used.

    Only ‘breslow’ ties are available; the literature is currently unclear on
    how to add stratification to Simon’s algorithm.

    Parameters
    ----------
    alpha : float, default=0.0
        Constant that multiplies the penalty terms. Used to penalize
        coefficients durring training.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    max_iter : int, default=100
        The maximum number of iterations.

    tol : float, default=1e-9
        The tolerance for the optimization: if the updates are smaller or equal
        to ``tol``, the optimization code checks the dual gap for optimality
        and continues until it is smaller or equal to ``tol``.

    References
    ----------
    [1] Simon N, Friedman J, Hastie T, Tibshirani R. Regularization Paths for
    Cox's Proportional Hazards Model via Coordinate Descent. J Stat Softw. 2011
    Mar;39(5):1-13. doi: 10.18637/jss.v039.i05. PMID: 27065756; PMCID:
    PMC4824408.
    """

    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        *,
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        max_iter: Optional[int] = 100,
        tol: float = 1e-9,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        times: np.ndarray[tuple[int], np.dtype[np.int64]],
        events: np.ndarray[tuple[int], np.dtype[np.bool_]],
        check_input: bool = True,
        times_start: Optional[np.ndarray[tuple[int], np.dtype[np.int64]]] = None,
    ):
        """
        Fit model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        times : array-like of shape (n_samples), dtype=np.int64
            Point in time last observed.

        events : array-like of shape (n_samples), dtype=np.bool_
            Experianed event.

        check_input : bool, default=True
            If True, validates and casts inputs.

        times_start : array-like of shape (n_samples, dtype=np.int64), default=None
            Starting point for observation. If not passed in, all times_start
            times are assumed to be 0.

        Returns
        -------
        object
            Fitted Estimator.
        """
        use_left_censorship = times_start is not None

        if check_input:
            X, times, events = validate_survival_data(X, times, events)
            if use_left_censorship:
                times_start = validate_times_start_array(times_start, times)

        self._max_time_observed = np.max(times)

        if use_left_censorship:
            coefs, loss = train_cox_elastic_net_with_left_censorship(
                X,
                times,
                times_start,
                events,
                self.alpha,
                self.l1_ratio,
                self.tol,
                self.max_iter,
            )
        else:
            coefs, loss = train_cox_elastic_net_regularization_paths(
                X, times, events, self.alpha, self.l1_ratio, self.tol, self.max_iter
            )

        self.coef_ = coefs
        self.n_log_likelihood = loss

        risk = np.exp(np.dot(X, self.coef_))
        self._breslow_base_hazard = _get_breslow_base_hazard(
            risk, times, events, self._max_time_observed
        )
        self._breslow_base_survival = np.exp(-self._breslow_base_hazard.cumsum())

        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        max_time: Optional[int] = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        Build survival curves on an array of vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predicting data.

        max_time : int, default=None
            Maximum time of built survival curves. If none, maximum time is max
            time seen on training data.

        Returns
        -------
        ndarray of shape (n_samples, max_time), dtype=np.float64
            The estimated survival curves, the left-most column is the
            probability of survival at time 1, and the right-most column ends
            at max_time.
        """

        check_is_fitted(self)

        if max_time is None:
            max_time = self._max_time_observed
        else:
            max_time = _as_int(max_time, "max_time")

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

    def predict_risk(
        self, X: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Build relative risk on an array of vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predicting data.

        Returns
        -------
        ndarray of shape (n_samples), dtype=np.float64
            The Relative risk of X, used under the hood for building survival
            curves. Relative risk is what 'Concordance Index' examines.
        """

        check_is_fitted(self)

        return np.exp(np.dot(X, self.coef_))
