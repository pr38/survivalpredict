from inspect import signature
from itertools import islice
from numbers import Integral
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, _fit_context, clone
from sklearn.utils._param_validation import HasMethods, Hidden, Interval
from sklearn.utils.metadata_routing import get_routing_for_object
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted, check_memory

try:
    from sklearn.utils._repr_html import _VisualBlock
except ImportError:
    try:
        from sklearn.utils._estimator_html_repr import _VisualBlock
    except ImportError:
        from sklearn.utils._repr_html.estimator import _VisualBlock

from ._data_validation import (
    _as_bool_np_array,
    _as_int,
    _as_int_np_array,
    validate_times_start_array,
    validate_times_array,
)
from ._estimator_utils import _get_estimator_names, _unpack_sklearn_pipeline_target
from .strata_preprocessing import StrataColumnTransformer

__all__ = [
    "build_sklearn_pipeline_target",
    "SklearnSurvivalPipeline",
    "make_sklearn_survival_pipeline",
]


def build_sklearn_pipeline_target(
    times: np.ndarray[tuple[int], np.dtype[np.integer]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    strata: Optional[np.ndarray[tuple[int], np.dtype[np.integer]]] = None,
    times_start: Optional[np.ndarray[tuple[int], np.dtype[np.integer]]] = None,
):
    """
    Target builder for survivalpredict's ‘SklearnSurvivalPipeline’.

    Takes ‘times’, ‘events’ arrays, and optionally ‘strata’ and ‘times_start’
    inputs; and builds a singular numpy array that can function as the
    ‘y’/observed for ‘SklearnSurvivalPipeline’ and scikit-learn's api.

    Parameters
    ----------
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
    ndarray
        Returns a numpy array that survivalpredict knows how to unpack, while
        allowing said numpy array to flow through the various machinery of
        scikit-learn.
    """
    times = validate_times_array(times)
    events = _as_bool_np_array(events)

    dtype = [("times", np.int64), ("events", bool)]
    if strata is not None:
        strata = _as_int_np_array(strata)
        dtype.append(("strata", np.int64))

    if times_start is not None:
        times_start = validate_times_start_array(times_start, times)
        dtype.append(("times_start", np.int64))

    y = np.empty(times.shape[0], dtype=dtype)

    y["times"] = times
    y["events"] = events

    if strata is not None:
        y["strata"] = strata

    if times_start is not None:
        y["times_start"] = times_start

    return y


def _fit_transform_sk_tranformer(
    trans: BaseEstimator, step_idx: int, X: Any
) -> tuple[BaseEstimator, Any]:
    X = trans.fit_transform(X)
    return trans, X


def _fit_transform_sp_strata_column_transformer(
    trans: StrataColumnTransformer,
    step_idx: int,
    X: Any,
    strata: Optional[np.ndarray[tuple[int], np.dtype[np.integer]]] = None,
) -> tuple[StrataColumnTransformer, Any, np.ndarray, np.ndarray]:
    X, _, _, strata = trans.fit_transform(X, strata=strata)
    return trans, X, strata


class SklearnSurvivalPipeline(_BaseComposition):
    """
    Scikit-learn compatible pipeline class for survivalpredict.

    A sequence of data transformers and strata preprocessing with a final
    predictor. Takes a feature matrix/X as well as the output
    ‘build_sklearn_pipeline_target’ as the ‘y’. Combined survivalpredict’s
    ‘sklearn_scorer’s, it allows users to build pipelines that can interface
    with the rest of Scikit-learn’s api. Parameters of the various steps using
    their names and the parameter name separated by a '__', allowing for
    parameters of various steps to be tuned during cross-validation searches.

    Parameters
    ----------
    steps : list[tuple[str, BaseEstimator]]
        List of the tuples with names and class instances that are chained
        together. The class instances are assumped to be scikit-learn
        transformers/survivalpredict StrataBuilders/StrataColumnTransformers.
        The final instance is assumed to be a survivalpredict estimator
        predictor.

    max_time : int
        Maximum time for building survival curves.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. The last step
        will never be cached, even if it is a transformer. By default, no
        caching is performed. If a string is given, it is the path to the
        caching directory. Enabling caching triggers a clone of the
        transformers before fitting. Therefore, the transformer instance given
        to the pipeline cannot be inspected directly. Use the attribute
        named_steps or steps to inspect estimators within the pipeline. Caching
        the transformers is advantageous when fitting is time consuming.
    """
    _parameter_constraints: dict = {
        "steps": [list, Hidden(tuple)],
        "max_time": [Interval(Integral, 1, None, closed="left")],
        "memory": [None, str, HasMethods(["cache"])],
    }

    def __init__(
        self,
        steps: list[tuple[str, BaseEstimator]],
        max_time: int,
        *,
        memory=None,
    ):
        self.steps = steps
        self.max_time = max_time
        self.memory = memory

    def _iter(self, with_final=True):
        stop = len(self.steps)
        if not with_final:
            stop -= 1

        for idx, (name, trans) in enumerate(islice(self.steps, 0, stop)):
            yield idx, name, trans

    @property
    def _final_estimator(self):

        return self.steps[-1][1]

    def _run_transformers(self, X, strata):

        memory = check_memory(self.memory)

        _fit_transform_sk_tranformer_cache = memory.cache(_fit_transform_sk_tranformer)
        _fit_transform_sp_strata_column_transformer_cache = memory.cache(
            _fit_transform_sp_strata_column_transformer
        )

        for idx, name, trans in self._iter(with_final=False):

            if hasattr(memory, "location") and memory.location is None:
                trans = trans
            else:
                trans = clone(trans)

            if isinstance(trans, StrataColumnTransformer):
                fitted_trans, X, strata = (
                    _fit_transform_sp_strata_column_transformer_cache(
                        trans, idx, X, strata
                    )
                )

            else:  # scikit learn transformers
                fitted_trans, X = _fit_transform_sk_tranformer_cache(trans, idx, X)

            self.steps[idx] = (name, fitted_trans)

        return X, strata

    def _fit(self, X, times, events, strata, times_start):

        if strata is not None:
            self._uses_strata_in_input = True
        else:
            self._uses_strata_in_input = False

        X, strata = self._run_transformers(X, strata)

        fit_params = {}

        if strata is not None:
            fit_params["strata"] = strata
        if times_start is not None:
            fit_params["times_start"] = times_start

        self._final_estimator.fit(X, times, events, **fit_params)

        return X, times, events, strata

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape
            Target values. Assumes that output of ‘build_sklearn_pipeline_target’.

        Returns
        -------
        object
            Returns the instance itself.
        """

        times, events, strata, times_start = _unpack_sklearn_pipeline_target(y)

        self._fit(X, times, events, strata, times_start)

        self.is_fitted_ = True

        return self

    def predict(self, X, strata=None):
        """
        Predict using the pipeline.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        strata : array-like of shape (n_samples,), dtype=np.int64, default=None
            If y from training/fit had prebuilt strata; strata can be passed into fit.

        Returns
        -------
        ndarray of shape (n_samples, max_time), dtype=np.float64
            The estimated survival curves, the left-most column is the
            probability of survival at time 1, and the right-most column ends
            at max_time.
        """

        check_is_fitted(self)

        if self._uses_strata_in_input and strata is None:
            raise ValueError(
                "strata must be present if pipeline is fit with preexisting strata"
            )

        max_time = _as_int(self.max_time, "max_time")

        X, strata = self._run_transformers(X, strata)

        if strata is not None:
            return self._final_estimator.predict(X, max_time=max_time, strata=strata)
        else:
            return self._final_estimator.predict(X, max_time=max_time)

    def fit_predict(self, X, y):
        times, events, strata, times_start = _unpack_sklearn_pipeline_target(y)

        max_time = _as_int(self.max_time, "max_time")

        X, times, events, strata = self._fit(X, times, events, strata, times_start)

        self.is_fitted_ = True

        if strata is not None:
            return self._final_estimator.predict(X, max_time=max_time, strata=strata)
        else:
            return self._final_estimator.predict(X, max_time=max_time)

    def get_params(self, deep=True):
        return self._get_params("steps", deep=deep)

    def set_params(self, **kwargs):
        self._set_params("steps", **kwargs)
        return self

    def _validate_steps(self):
        if not self.steps or len(self.steps) == 0:
            raise ValueError("The pipeline is empty. Please add steps.")
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
                t, "transform"
            ):
                raise TypeError(
                    "All intermediate steps should be "
                    "transformers and implement fit and transform "
                    "'%s' (type %s) doesn't" % (t, type(t))
                )

        fit_arg_keys = list(signature(estimator.fit).parameters.keys())
        is_compatiable = ["X", "times", "events"] == fit_arg_keys[:3]

        if (
            not hasattr(estimator, "fit")
            or not hasattr(estimator, "predict")
            or not is_compatiable
        ):
            raise TypeError(
                "Last step of Pipeline should implement fit and predict"
                "and accept 'X','times','events' as inputs "
                "'%s' (type %s) doesn't" % (estimator, type(estimator))
            )

    def _sk_visual_block_(self):

        names, estimators = zip(*self.steps)
        name_details = str(estimators)
        return _VisualBlock(
            "serial",
            estimators,
            names=names,
            name_details=name_details,
            dash_wrapped=False,
        )


def make_sklearn_survival_pipeline(
    *steps_no_names: BaseEstimator, max_time: int, memory=None
):
    """
    Construct a SklearnSurvivalPipeline from given steps.

    This is a shorthand for the SklearnSurvivalPipeline constructor; it does
    not require, and does not permit, naming the steps. Instead, their names
    will be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps_no_names : list of Estimator objects 
        List of class instances that are chained together. The class instances 
        are assumped to be scikit-learn transformers/survivalpredict 
        StrataBuilders/StrataColumnTransformers. The final instance is assumed to be 
        a survivalpredict estimator predictor.

    max_time : int
        Maximum time for building survival curves.

    memory : str or object with the joblib.Memory interface, default=None Used
        Used to cache the fitted transformers of the pipeline. The last step will never
        be cached, even if it is a transformer. By default, no caching is
        performed. If a string is given, it is the path to the caching directory.
        Enabling caching triggers a clone of the transformers before fitting.
        Therefore, the transformer instance given to the pipeline cannot be
        inspected directly. Use the attribute named_steps or steps to inspect
        estimators within the pipeline. Caching the transformers is advantageous
        when fitting is time consuming.
    """
    max_time = _as_int(max_time, "max_time")

    names = _get_estimator_names(steps_no_names)
    steps_with_names = list(zip(names, steps_no_names))

    return SklearnSurvivalPipeline(steps_with_names, max_time, memory=memory)
