from copy import deepcopy
from itertools import product
from numbers import Integral
from typing import Any, Literal, Optional, Protocol, Sequence

import numba as nb
import numpy as np
from scipy.cluster.vq import kmeans
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context, clone
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted

try:
    from sklearn.utils._repr_html import _VisualBlock
except ImportError:
    try:
        from sklearn.utils._estimator_html_repr import _VisualBlock
    except ImportError:
        from sklearn.utils._repr_html.estimator import _VisualBlock

from ._data_validation import _as_int_np_array, _as_numeric_np_array
from ._estimator_utils import _get_estimator_names

__all__ = [
    "StrataBuilderDiscretizer",
    "StrataBuilderEncoder",
    "StrataColumnTransformer",
    "make_strata_column_transformer",
]


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


class StrataBuilderDiscretizer(_StrataBuilderBase, auto_wrap_output_keys=None):
    """
    Builds strata keys from numeric data.

    If predefined 'splits' are given, strata are built via the given bins
    and 'n_splits' and 'strategy' are ignored. Otherwise 'n_splits' and
    'strategy' is used to generate bins. Largely inspired by
    scikitlearn's KBinsDiscretizer.Adds onto existing strata, if
    existing strata are passed in.

    Parameters
    ----------
    n_bins : int , default=5
        The number of bins to produce. Raises ValueError if ``n_bins < 2``.

        'n_bins' is ignored if 'splits' is not None.

    strategy : {'uniform','quantile','kmeans'}, default='quantile'
        Strategy used to define the widths of the bins.

        - 'uniform': All bins in each feature have identical widths.
        - 'quantile': All bins in each feature have the same number of points.
        - 'kmeans': Values in each bin have the same nearest center of a 1D
          k-means cluster.

        'strategy' is ignored if 'splits' is not None.

    splits : numeric array-like, default=None
        Predefined splits to build bins.
        If 'splits' is None, strategy and n_bins is ignored.

    Attributes
    ----------
    _splits : ndarray of ndarray of shape (n_features,)
        Splits used to generate bins.

    _uses_strata : bool
        True if fitted on preexising strata, False otherwise.
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
        "n_bins": [
            Interval(Integral, 2, None, closed="left"),
        ],
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

    def _find_splits(self, X):

        if self.strategy == "uniform":
            col_mins = X.min(axis=0)
            col_maxs = X.max(axis=0)
            bins_np = np.linspace(col_mins, col_maxs, self.n_bins, axis=1)
            splits_np = bins_np  # [:, 1:-1]
            splits = [np.unique(i) for i in splits_np]

        elif self.strategy == "quantile":
            percentile_levels = np.linspace(0, 100, self.n_bins)
            bins_np = np.percentile(X, percentile_levels, axis=0).T
            splits_np = bins_np  # [:, 1:-1]
            splits = [np.unique(i) for i in splits_np]

        else:  # self.strategy == 'kmeans':

            splits = []

            for i_col in range(X.shape[1]):
                Xi = X[:, i_col]
                splits_i, _ = kmeans(Xi, self.n_bins)
                splits.append(np.sort(splits_i))

        return splits

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, times=None, events=None, strata=None, check_input=True):
        """
        Learn the strata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.

        times : array-like of shape n_samples, default=None
            Ignored.

        events : array-like of shape n_samples, default=None
            Ignored.

        strata : array-like of shape n_samples, default=None
            Preexsting strata, the strata built will add onto the preexsting strata.

        check_input : bool, default True
            If True, runs checks and casting on data to ensure data is valid.
        
        Returns
        -------
        object
            Returns the instance itself.
        """
        self._uses_strata = strata is not None

        if check_input == True and self._uses_strata:
            strata = _as_int_np_array(strata, "strata")

            if len(strata.shape) > 1:
                raise ValueError("strata must be 1 dimensional")

        X = _as_numeric_np_array(X)

        if len(X.shape) == 1:
            X = X[:, None]

        self._n_cols_seen = X.shape[1]

        self._n_strata_cols = X.shape[1]

        if self.splits is None:
            splits = self._find_splits(X)
        else:
            splits = self.splits

        self._splits = splits

        if self._uses_strata:
            self.preexisting_strata_keys = np.unique(strata)

            possible_splits = [strata] + self._splits

        else:
            possible_splits = self._splits

        posible_digitized_keys = [
            tuple(i) for i in product(*(range(len(i) + 2) for i in possible_splits))
        ]

        self._digitized_map = dict(
            zip(posible_digitized_keys, range(len(posible_digitized_keys)))
        )

        self.is_fitted_ = True

        return self

    def transform(self, X, times=None, events=None, strata=None):
        """
        Discretize numerical data to build strata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.

        times : array-like of shape n_samples, default=None
            Ignored.

        events : array-like of shape n_samples, default=None
            Ignored.

        strata : array-like of shape n_samples, default=None
            Preexsting strata, the strata built will add onto the preexsting strata.

        Returns
        -------
        array-like of shape (n_samples) , dtype=np.int64
            Build strata.
        """

        check_is_fitted(self)

        if self._uses_strata:
            if strata is None:
                raise ValueError(
                    "strata must be present if transformer is fitted with strata"
                )
            strata = _as_int_np_array(strata, "strata")

            if len(strata.shape) > 1:
                raise ValueError("strata must be 1 dimensional")

        X = np.array(X)

        if len(X.shape) == 1:
            X = X[:, None]

        X = X.astype(np.float64, order="C")

        if X.shape[1] != self._n_cols_seen:
            raise ValueError(
                "number of columns must match data used to fit this transformer "
            )

        digitized = digitized_per_col(X, self._splits)

        if self._uses_strata:
            digitized = np.hstack((strata[:, None], digitized))

        return np.array([self._digitized_map[tuple(i)] for i in digitized.tolist()])

    def fit_transform(self, X, times=None, events=None, strata=None):
        """
        Fit and build strata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.

        times : array-like of shape n_samples, default=None
            Ignored.

        events : array-like of shape n_samples, default=None
            Ignored.

        strata : array-like of shape n_samples, default=None
            Preexsting strata, the strata built will add onto the preexsting strata.

        Returns
        -------
        ndarray of shape (n_samples) , dtype=np.int64
            Build strata.
        """

        self.fit(X, times=None, events=None, strata=strata)

        return self.transform(X, times=None, events=None, strata=strata)


class StrataBuilderEncoder(_StrataBuilderBase, auto_wrap_output_keys=None):
    """
    Builds strata keys from categorical data.

    If existing strata are passed in, it adds onto existing strata.
    StrataBuilderEncoder works on categorical data encoded in numerical or string types.
    One or many columns of mixed types can be used.
    """

    def fit(self, X, times=None, events=None, strata=None, check_input=True):
        """
        Learn the strata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.

        times : array-like of shape n_samples, default=None
            Ignored.

        events : array-like of shape n_samples, default=None
            Ignored.

        strata : array-like of shape n_samples, default=None
            Preexsting strata, the strata built will add onto the preexsting strata.

        check_input : bool, default True
            If True, runs checks and casting on data to ensure data is valid.

        Returns
        -------
        object
            Returns the instance itself.
        """

        self._uses_strata = strata is not None

        if check_input == True and self._uses_strata:
            strata = _as_int_np_array(strata, "strata")

            if len(strata.shape) > 1:
                raise ValueError("strata must be 1 dimensional")

        X = np.array(X)

        if len(X.shape) == 1:
            X = X[:, None]

        self._n_cols_seen = X.shape[1]

        if self._uses_strata:
            if len(X.shape) == 1:
                X = X[:, None]

            X = np.hstack((strata[:, None], X))

        self.strata_keys_ = np.unique(X, axis=0)
        if len(self.strata_keys_.shape) == 1:
            self.strata_keys_ = self.strata_keys_[:, None]

        self.strata_to_key_map = dict(
            zip(
                [tuple(i) for i in self.strata_keys_.tolist()],
                range(len(self.strata_keys_)),
            )
        )

        self.is_fitted_ = True

        return self

    def transform(self, X, times=None, events=None, strata=None):
        """
        Encode categorical data to build strata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be encoded.

        times : array-like of shape n_samples, default=None
            Ignored.

        events : array-like of shape n_samples, default=None
            Ignored.

        strata : array-like of shape n_samples, default=None
            Preexsting strata, the strata built will add onto the preexsting strata.

        Returns
        -------
        ndarray of shape (n_samples)  , dtype=np.int64
            Build strata.
        """
        check_is_fitted(self)

        if self._uses_strata:
            if strata is None:
                raise ValueError(
                    "strata must be present if transformer is fitted with strata"
                )
            strata = _as_int_np_array(strata, "strata")

            if len(strata.shape) > 1:
                raise ValueError("strata must be 1 dimensional")

        X = np.array(X)

        if len(X.shape) == 1:
            X = X[:, None]

        if X.shape[1] != self._n_cols_seen:
            raise ValueError(
                "number of columns must match data used to fit this transformer "
            )

        if self._uses_strata:
            X = np.hstack((strata[:, None], X))

        return np.array([self.strata_to_key_map[tuple(i)] for i in X.tolist()])

    def fit_transform(self, X, times=None, events=None, strata=None):
        """
        Fit and build strata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be encoded.

        times : array-like of shape n_samples, default=None
            Ignored.

        events : array-like of shape n_samples, default=None
            Ignored.

        strata : array-like of shape n_samples, default=None
            Preexsting strata, the strata built will add onto the preexsting strata.

        Returns
        -------
        ndarray of shape (n_samples) , dtype=np.int64
            Build strata.
        """

        self._uses_strata = strata is not None

        if self._uses_strata:
            strata = _as_int_np_array(strata, "strata")

            if len(strata.shape) > 1:
                raise ValueError("strata must be 1 dimensional")

        X = np.array(X)

        if self._uses_strata:
            if len(X.shape) == 1:
                X = X[:, None]

            X = np.hstack((strata[:, None], X))

        if len(X.shape) == 1:
            self._n_cols_seen = 1
        else:
            self._n_cols_seen = X.shape[1]

        # the double unique trick allows us to deal with numpy object/string types with axis
        _, _, return_inverse1 = np.unique(X, return_index=True, return_inverse=True)
        _, return_index2, new_strata = np.unique(
            return_inverse1, return_index=True, return_inverse=True, axis=0
        )

        self.strata_keys_ = X[return_index2]

        self.strata_to_key_map = dict(
            zip(
                [tuple(i) for i in self.strata_keys_.tolist()],
                range(len(self.strata_keys_)),
            )
        )

        self.is_fitted_ = True

        return new_strata


class StrataBuilderProtocal(Protocol):
    def fit(self, X, times=None, events=None, strata=None):
        pass

    def transform(
        self, X, times=None, events=None, strata=None
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
        pass

    def fit_transform(
        self, X, times=None, events=None, strata=None
    ) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
        pass


class StrataColumnTransformer(
    TransformerMixin, _BaseComposition, auto_wrap_output_keys=None
):
    """
    Applies StrataBuilders to columns of an array or DataFrame.

    Functions much like scikit-learn's ColumnTransformer class, but for survivalpredict's
    StrataBuilders instead of scikit-learn's Transformers. Different columns or column subsets
    of the input are separately run through different StrataBuilders. If there are pre-existing strata,
    it will be added to the created strata. After the strata are built, columns used for building said strata are
    then removed from the feature set. Works on Numpy arrays as well as  Pandas/Polars dataframes.

    Designed to be used with 'survivalpredict.pipeline.SklearnSurvivalPipeline'.
    The columns in strata_transformers tuples are exposed as parameters that can be tuned.
    This is useful in model selection, where different strata are tried. Said prams are named as such '{name}_columns'.

    Parameters
    ----------
    strata_transformers : list of tuples
        List of (name, strat builder object, columns) tuples specifying the name, strat builder object and columns  used for building the strata.
    """

    _parameter_constraints: dict = {"strata_transformers": [list]}

    def __init__(
        self,
        strata_transformers: list[
            tuple[str, StrataBuilderProtocal, Sequence[Any] | int | str | slice]
        ],
    ):
        self.strata_transformers = strata_transformers

    def _get_columns_from_selections(self, selections: Sequence, n_cols: int) -> set:
        columns = set()

        if type(selections) in (int, float, str):
            columns.add(selections)

        else:

            for s in selections:
                if type(s) == slice:
                    for c in s.indices(n_cols):
                        columns.add(c)
                elif type(s) in (int, float, str):
                    columns.add(s)
                elif type(s) == list or type(s) == tuple or hasattr(s, "__iter__"):
                    for c in s:
                        columns.add(c)
                else:
                    columns.add(s)

        return columns

    def _get_selected_subset_of_data(self, X, selections):
        if callable(selections):  # selection is callable
            return selections(X)
        elif hasattr(X, "__dataframe__"):  # if X is pandas or polars dataframe
            n_cols = len(X.columns)
            columns = self._get_columns_from_selections(selections, n_cols)
            columns_as_list = list(columns)
            return X[columns_as_list]
        else:  # is numpy array or list
            X = np.array(X)
            n_cols = X.shape[1]
            columns = self._get_columns_from_selections(selections, n_cols)
            columns_as_list = list(columns)
            return X[:, columns_as_list]

    def _remove_selected_subset_of_data(self, X, selections):

        if hasattr(X, "__dataframe__"):
            n_cols = len(X.columns)
        elif type(X) == np.ndarray:
            n_cols = X.shape[1]
        else:  # assuming list
            X = np.array(X)
            n_cols = X.shape[1]

        columns = self._get_columns_from_selections(selections, n_cols)

        if type(X) == np.ndarray:
            if len(columns) > 0:
                columns_to_keep = set(range(X.shape[1])).difference(columns)
                X = X[:, list(columns_to_keep)]
        elif hasattr(X, "__dataframe__"):
            if len(columns) > 0:
                remaining_columns = set(X.columns)
                columns_to_keep = list(remaining_columns.difference(columns))
                X = X[columns_to_keep]

        return X

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, times=None, events=None, strata=None, check_input=True):
        """
        Learn the strata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dataset used for building strata.

        times : array-like of shape n_samples, default=None
            Ignored.

        events : array-like of shape n_samples, default=None
            Ignored.

        strata : array-like of shape n_samples, default=None
            Preexsting strata, the strata built will add onto the preexsting strata.

        check_input : bool, default True
            If True, runs checks and casting on data to ensure data is valid.

        Returns
        -------
        self
            Returns the instance itself.
        """

        self._validate_strata_transformers()

        self.has_preexisting_strata = strata is not None

        if check_input == True and self.has_preexisting_strata:
            strata = _as_int_np_array(strata, "strata")

            if len(strata.shape) > 1:
                raise ValueError("strata must be 1 dimensional")

        for _, st, selection in self.strata_transformers:
            X_ = self._get_selected_subset_of_data(X, selection)
            st.fit(X_, times, events)

        self.is_fitted_ = True

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, times=None, events=None, strata=None, check_input=True):
        """
        Fit and build strata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.

        times : array-like of shape n_samples, default=None
            Ignored.

        events : array-like of shape n_samples, default=None
            Ignored.

        strata : array-like of shape n_samples, default=None
            Preexsting strata, the strata built will add onto the preexsting strata.

        check_input : bool, default True
            If True, runs checks and casting on data to ensure data is valid.

        Returns
        -------
        X : ndarray of shape (n_samples, ???)
            The feature set without the columns used for strata.

        times : ndarray of shape (n_samples)
            The same times array passed into method.

        events : ndarray of shape (n_samples)
            The same events array passed into method.

        strata : ndarray of shape (n_samples) , dtype=np.int64
            The strata build from the original feature set.
        """


        self._validate_strata_transformers()

        self.has_preexisting_strata = strata is not None

        selections = []

        if strata is not None:
            strata = _as_int_np_array(strata, "strata")
            if len(strata.shape) > 1:
                raise ValueError("strata must be 1 dimensional")

            self.has_preexisting_strata = True
            stratas = [strata]
        else:
            self.has_preexisting_strata = False
            stratas = []

        for _, st, selection in self.strata_transformers:
            selections.append(selection)
            X_ = self._get_selected_subset_of_data(X, selection)
            strata = st.fit_transform(X_, times, events, strata)
            stratas.append(strata)

        self._meta_keys, strata = np.unique(stratas, axis=1, return_inverse=True)

        X = self._remove_selected_subset_of_data(X, selections)

        self.is_fitted_ = True

        return X, times, events, strata

    def transform(self, X, times=None, events=None, strata=None, check_input=True):
        """
        Build strata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dataset used for building strata.

        times : array-like of shape n_samples, default=None
            Ignored.

        events : array-like of shape n_samples, default=None
            Ignored.

        strata : array-like of shape n_samples, default=None
            Preexsting strata, the strata built will add onto the preexsting strata.

        check_input : bool, default True
            If True, runs checks and casting on data to ensure data is valid.

        Returns
        -------
        X : ndarray of shape (n_samples, ???)
            The feature set without the columns used for strata.

        times : ndarray of shape (n_samples)
            The same times array passed into method.

        events : ndarray of shape (n_samples)
            The same events array passed into method.

        strata : ndarray of shape (n_samples) , dtype=np.int64
            The strata build from the original feature set.
        """

        check_is_fitted(self)

        selections = []

        if self.has_preexisting_strata:
            if strata is None:
                raise ValueError(
                    "strata must be present if transformer is fitted with strata"
                )
            strata = _as_int_np_array(strata, "strata")

            if len(strata.shape) > 1:
                raise ValueError("strata must be 1 dimensional")

        if strata is not None:
            stratas = [strata]
        else:
            stratas = []

        for _, st, selection in self.strata_transformers:
            selections.append(selection)
            X_ = self._get_selected_subset_of_data(X, selection)
            strata = st.transform(X_, times, events, strata)
            stratas.append(strata)

        self._meta_keys, strata = np.unique(stratas, axis=1, return_inverse=True)

        X = self._remove_selected_subset_of_data(X, selections)

        return X, times, events, strata

    def get_params(self, deep=True):
        params = {}
        params["strata_transformers"] = self.strata_transformers

        for item_name, tranfomer, columns in self.strata_transformers:
            for pram_key, pram_value in tranfomer.get_params(deep=True).items():
                params["%s__%s" % (item_name, pram_key)] = pram_value
            params["%s__%s" % (item_name, "columns")] = columns

        return params

    def set_params(self, **kwargs):

        if "strata_transformers" in kwargs:
            self.strata_transformers = kwargs["strata_transformers"]

        item_names, _, _ = zip(*self.strata_transformers)

        item_name_columns = [
            "%s__%s" % (item_name, "columns") for item_name in item_names
        ]

        for name in list(kwargs.keys()):
            if "__" not in name and name in item_names:
                self._replace_estimator("strata_transformers", name, kwargs.pop(name))

            elif name in item_name_columns:
                index = item_name_columns.index(name)
                item_name, tranfomer, _ = self.strata_transformers[index]
                new_columns = kwargs.pop(name)
                self.strata_transformers[index] = (item_name, tranfomer, new_columns)

        super().set_params(**kwargs)

        return self

    def _sk_visual_block_(self):

        names, transformers, name_details = zip(*self.strata_transformers)
        return _VisualBlock(
            "parallel",
            transformers,
            names=names,
            name_details=name_details,
            dash_wrapped=False,
        )

    @property
    def _strata_transformers(self):
        try:
            return [(name, trans, sel) for name, trans, sel in self.strata_transformers]
        except (TypeError, ValueError):
            return self.strata_transformers

    @_strata_transformers.setter
    def _strata_transformers(self, value):
        try:
            self.strata_transformers = [
                (name, clone(trans), sel) for (name, trans, sel) in value
            ]
        except (TypeError, ValueError):
            self.strata_transformers = value

    def _validate_strata_transformers(self):

        names, transformers, _ = zip(*self.strata_transformers)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if not (hasattr(t, "fit") and hasattr(t, "fit_transform")) and not hasattr(
                t, "transform"
            ):
                raise TypeError(
                    "All strata transformers should implement fit, transform and fit_transform"
                )

    def __sklearn_clone__(self):

        strata_transformers = [
            (deepcopy(name), clone(trans), deepcopy(sel))
            for (name, trans, sel) in self.strata_transformers
        ]
        return StrataColumnTransformer(strata_transformers)


def make_strata_column_transformer(
    *strata_transformer_columns: tuple[
        StrataBuilderProtocal, Sequence[Any] | int | str | slice
    ],
) -> StrataColumnTransformer:
    """
    Construct a StrataColumnTransformer from the given transformers and columns.

    Functions much like scikit-learn's make_column_transformer class, but for survivalpredict's
    StrataBuilders instead of scikit-learn's Transformers. A utility for building StrataColumnTransformer without naming strata builders.

    Parameters
    ----------
    strata_transformers : list of tuples
        List of (strat builder object, columns) tuples specifying strat builder object and columns used for building the strata.

    Returns
    -------
    StrataColumnTransformer
            Returns a StrataColumnTransformer object.
    """
    estimators, selections = zip(*strata_transformer_columns)

    names = _get_estimator_names(estimators)

    strata_transformers = list(zip(names, estimators, selections))

    return StrataColumnTransformer(strata_transformers)
