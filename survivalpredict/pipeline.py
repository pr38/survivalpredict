from collections import Counter
from inspect import signature
from itertools import islice
from numbers import Integral
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.base import BaseEstimator, _fit_context, clone
from sklearn.utils._param_validation import HasMethods, Hidden, Interval
from sklearn.utils.metadata_routing import get_routing_for_object
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted, check_memory

try:
    from sklearn.utils._repr_html import _VisualBlock
except ImportError:
    from sklearn.utils._estimator_html_repr import _VisualBlock

from ._data_validation import _as_bool_np_array, _as_int, _as_int_np_array
from ._estimator_utils import _get_estimator_names
from .strata_preprocessing import StrataColumnTransformer

__all__ = [
    "SklearnSurvivalPipeline",
    "make_sklearn_survival_pipeline",
]


def build_sklearn_pipeline_target(times, events, strata=None):
    times = _as_int_np_array(times)
    events = _as_bool_np_array(events)

    dtype = [("times", np.int64), ("events", bool)]

    if strata is not None:
        strata = _as_int_np_array(strata)
        dtype.append(("strata", np.int64))
        y = np.empty(times.shape[0], dtype=dtype)
        y["strata"] = strata
    else:
        y = np.empty(times.shape[0], dtype=dtype)

    y["times"] = times
    y["events"] = events

    return y


def _unpack__sklearn_pipeline_target(
    y: np.ndarray,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.integer]],
    np.ndarray[tuple[int], np.dtype[np.bool_]],
    Optional[np.ndarray[tuple[int], np.dtype[np.integer]]],
]:
    times = _as_int_np_array(y["times"])
    events = _as_bool_np_array(y["events"])

    if "strata" in y.dtype.names:
        strata = _as_int_np_array(y["strata"], "strata")
        return times, events, strata
    else:
        return times, events, None


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
    _parameter_constraints: dict = {
        "steps": [list, Hidden(tuple)],
        "max_time": [Interval(Integral, 1, None, closed="left")],
        "transform_input": [list, None],
        "memory": [None, str, HasMethods(["cache"])],
    }

    def __init__(
        self,
        steps: list[tuple[str, BaseEstimator]],
        max_time: int,
        *,
        transform_input=None,
        memory=None,
    ):
        self.steps = steps
        self.max_time = max_time
        self.transform_input = transform_input
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

    def _fit(self, X, times, events, strata):

        if strata is not None:
            self._uses_strata_in_input = True
        else:
            self._uses_strata_in_input = False

        X, strata = self._run_transformers(X, strata)

        if strata is not None:
            self._final_estimator.fit(
                X,
                times,
                events,
                strata=strata,
            )
        else:
            self._final_estimator.fit(X, times, events)

        self.is_fitted_ = True

        return X, times, events, strata

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        times, events, strata = _unpack__sklearn_pipeline_target(y)

        self._fit(X, times, events, strata)

        return self

    def predict(self, X, strata=None):
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
        times, events, strata = _unpack__sklearn_pipeline_target(y)

        max_time = _as_int(self.max_time, "max_time")

        X, times, events, strata = self._fit(X, times, events, strata)

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
    *steps_no_names: BaseEstimator, max_time: int, transform_input=None, memory=None
):
    max_time = _as_int(max_time, "max_time")

    names = _get_estimator_names(steps_no_names)
    steps_with_names = list(zip(names, steps_no_names))

    return SklearnSurvivalPipeline(
        steps_with_names, max_time, transform_input=transform_input, memory=memory
    )
