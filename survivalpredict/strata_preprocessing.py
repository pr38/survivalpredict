from itertools import product
from numbers import Integral, Real
from typing import Literal, Optional

import numba as nb
import numpy as np
from scipy.cluster.vq import kmeans
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ._data_validation import _as_int_np_array, _as_numeric_np_array

__all__ = ["StrataDiscretizer"]


digitized_per_col_signature = nb.types.Array(
    nb.types.int64, 2, "C", False, aligned=True
)(
    nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
    nb.types.List(nb.types.Array(nb.types.float64, 1, "C"), True),
)


@nb.njit(digitized_per_col_signature, cache=True)
def digitized_per_col(X, splits):

    n_cols = X.shape[1]

    digitized_out = np.empty(X.shape, dtype=np.int64)

    for col_i in range(n_cols):
        col = X[:, col_i]
        s = splits[col_i]

        digitized = np.digitize(col, s, right=True)
        digitized_out[:, col_i] = digitized

    return digitized_out


class _StrataBuilderBase(TransformerMixin, BaseEstimator):
    pass


class StrataBuilderDiscretizer(_StrataBuilderBase):
    """
    Builds strata keys from numeric data. Add onto existing strata, if existing strata is passed in.

    if predefined 'splits' are given, strata is build via the given bins and 'n_splits' and 'strategy' is ignored.
    Otherwise 'n_splits' and 'strategy' is used to generate bins.
    Largly inspired by scikitlearn's KBinsDiscretizer.

    """

    _parameter_constraints: dict = {
        "strategy": [
            StrOptions(
                {
                    "uniform",
                    "quantile",
                    "kmeans",
                }
            )
        ],
        "n_bins": [Interval(Integral, 2, None, closed="left")],
        "splits": ["array-like", None],
    }

    def __init__(
        self,
        n_bins: Optional[int] = 5,
        strategy: Literal["uniform", "quantile", "kmeans"] = "quantile",
        splits: (
            list[float | int]
            | list[list[float | int]]
            | np.ndarray[
                tuple[int] | tuple[int, int], np.dtype[np.floating | np.integer]
            ]
            | None
        ) = None,
    ):
        self.n_bins = n_bins
        self.strategy = strategy
        self.splits = splits

        self.is_fitted_ = False

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, times=None, events=None, strata=None, check_input=True):

        self._uses_strata = strata is not None

        if check_input == True and self._uses_strata:
            strata = _as_int_np_array(strata, "strata")

            if len(strata.shape) > 1:
                raise ValueError("strata must be 1 dimensional")

        X = _as_numeric_np_array(X)

        if len(X.shape) == 1:
            X = X[:, None]

        self._n_strata_cols = X.shape[1]

        if self.splits is None:
            if self.strategy == "uniform":
                col_mins = X.min(axis=0)
                col_maxs = X.max(axis=0)
                bins_np = np.linspace(col_mins, col_maxs, self.n_bins, axis=1)
                splits_np = bins_np[:, 1:-1]
                splits = [np.unique(i) for i in splits_np]

            elif self.strategy == "quantile":
                percentile_levels = np.linspace(0, 100, self.n_bins)
                bins_np = np.percentile(X, percentile_levels, axis=0).T
                splits_np = bins_np[:, 1:-1]
                splits = [np.unique(i) for i in splits_np]

            else:  # self.strategy == 'kmeans':

                splits = []

                for i_col in range(X.shape[1]):
                    Xi = X[:, i_col]
                    splits_i, _ = kmeans(Xi, self.n_bins - 2)
                    splits.append(np.sort(splits_i))

        else:
            splits = self.splits

        self._splits = splits

        self.is_fitted_ = True

        return self

    def transform(self, X, times=None, events=None, strata=None):

        check_is_fitted(self)

        if self._uses_strata:
            if strata is None:
                raise ValueError(
                    "strata must be present if model is trained with strata"
                )
            strata = _as_int_np_array(strata, "strata")

            if len(strata.shape) > 1:
                raise ValueError("strata must be 1 dimensional")

        if len(X.shape) == 1:
            X = X[:, None]

        X = X.astype(np.float64, order="C")

        digitized = digitized_per_col(X, self._splits)

        if self._uses_strata:
            digitized = np.hstack((strata[:, None], digitized))

        seen_strata_keys, new_strata = np.unique(digitized, return_inverse=True, axis=0)

        self._last_seen_strata_keys = seen_strata_keys

        return new_strata
