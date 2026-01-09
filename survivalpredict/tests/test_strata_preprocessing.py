import numpy as np
import pytest
import pandas as pd
import polars as pl

from ..strata_preprocessing import (
    StrataBuilderDiscretizer,
    StrataBuilderEncoder,
    StrataColumnTransformer,
)
from ..datasets import load_iranian_telecom_churn


@pytest.fixture
def X_times_events_column_names():
    data = load_iranian_telecom_churn()

    X = data["X"]
    times = data["times"]
    events = data["events"]
    column_names = data["column_names"]

    return X, times, events, column_names


def test_StrataBuilderEncoder(X_times_events_column_names):
    X, times, events, column_names = X_times_events_column_names

    position_of_categorical_cols = (
        np.argwhere(
            np.logical_or(column_names == "status", column_names == "tariff_plan")
        )
    ).flatten()

    # fit with 1 col
    a = StrataBuilderEncoder().fit(
        X[:, position_of_categorical_cols[0]],
        times,
        events,
    )
    a_r = a.transform(
        X[:, position_of_categorical_cols[0]],
        times,
        events,
    )
    assert len(a_r) == X.shape[0]
    assert len(a_r.shape) == 1
    assert a_r.dtype == np.int64

    # fit with 2 col
    b = StrataBuilderEncoder().fit(X[:, position_of_categorical_cols], times, events)
    b_r = b.transform(
        X[:, position_of_categorical_cols],
        times,
        events,
    )

    assert len(b_r) == X.shape[0]
    assert len(b_r.shape) == 1
    assert b_r.dtype == np.int64

    n_row = X.shape[0]
    n_cat1 = int(n_row / 2)
    n_cat2 = n_row - n_cat1

    string_col = np.array(
        ["monkey" for i in range(n_cat1)] + ["dog" for i in range(n_cat2)]
    )

    # fit with string col
    c = StrataBuilderEncoder().fit(string_col, times, events)
    c_r = c.transform(
        string_col,
        times,
        events,
    )

    assert len(c_r) == X.shape[0]
    assert len(c_r.shape) == 1
    assert c_r.dtype == np.int64

    string_cols = np.vstack((string_col, string_col)).T

    d = StrataBuilderEncoder().fit(string_cols, times, events)
    d_r = d.transform(
        string_cols,
        times,
        events,
    )

    assert len(d_r) == X.shape[0]
    assert len(d_r.shape) == 1
    assert d_r.dtype == np.int64


@pytest.mark.parametrize("n_bins", [2, 5, 10])
@pytest.mark.parametrize("strategy", ["quantile", "uniform", "kmeans"])
def test_StrataBuilderDiscretizer(X_times_events_column_names, n_bins, strategy):
    X, times, events, column_names = X_times_events_column_names

    position_of_numeric_cols = (
        np.argwhere(
            np.logical_or(
                column_names == "frequency_of_use", column_names == "charge_amount"
            )
        )
    ).flatten()

    # fit with 1 col
    a = StrataBuilderDiscretizer(n_bins=n_bins, strategy=strategy).fit(
        X[:, position_of_numeric_cols[0]],
        times,
        events,
    )
    a_r = a.transform(
        X[:, position_of_numeric_cols[0]],
        times,
        events,
    )

    assert len(a_r) == X.shape[0]
    assert len(a_r.shape) == 1
    assert a_r.dtype == np.int64

    # fit with 2 col
    b = StrataBuilderDiscretizer(n_bins=n_bins, strategy=strategy).fit(
        X[:, position_of_numeric_cols], times, events
    )
    b_r = b.transform(
        X[:, position_of_numeric_cols],
        times,
        events,
    )

    assert len(b_r) == X.shape[0]
    assert len(b_r.shape) == 1
    assert b_r.dtype == np.int64


def test_StrataColumnTransformer(X_times_events_column_names):
    X, times, events, column_names = X_times_events_column_names

    position_of_categorical_cols = (
        np.argwhere(
            np.logical_or(column_names == "status", column_names == "tariff_plan")
        )
    ).flatten()

    position_of_numeric_cols = (
        np.argwhere(
            np.logical_or(
                column_names == "frequency_of_use", column_names == "charge_amount"
            )
        )
    ).flatten()

    st = StrataColumnTransformer(
        [
            ("strataencoder", StrataBuilderEncoder(), position_of_categorical_cols),
            (
                "stratadiscretizer",
                StrataBuilderDiscretizer(n_bins=10),
                position_of_numeric_cols,
            ),
        ]
    )

    st.fit(X, times, events)
    X2, _, _, st_r = st.transform(X, times, events)

    assert X2.shape[1] == (
        X.shape[1]
        - (
            len(
                position_of_categorical_cols.tolist()
                + position_of_numeric_cols.tolist()
            )
        )
    )  # strata cols are removed
    assert len(st_r) == X.shape[0]
    assert len(st_r.shape) == 1
    assert st_r.dtype == np.int64

    st2 = StrataColumnTransformer(
        [
            ("strataencoder", StrataBuilderEncoder(), position_of_categorical_cols),
            (
                "stratadiscretizer",
                StrataBuilderDiscretizer(n_bins=10),
                position_of_numeric_cols,
            ),
        ]
    )

    preexisint_strata = np.random.randint(0, 5, X.shape[0])

    X3, _, _, st2_r = st2.fit_transform(X, times, events, strata=preexisint_strata)

    assert X3.shape[1] == (
        X.shape[1]
        - (
            len(
                position_of_categorical_cols.tolist()
                + position_of_numeric_cols.tolist()
            )
        )
    )  # strata cols are removed
    assert len(st2_r) == X.shape[0]
    assert len(st2_r.shape) == 1
    assert st2_r.dtype == np.int64


def test_StrataColumnTransformer_with_dataframes(X_times_events_column_names):
    X, times, events, column_names = X_times_events_column_names

    X_pd = pd.DataFrame(X, columns=column_names)

    X_pl = pl.from_pandas(X_pd)

    categorical_cols = ["status", "tariff_plan"]

    numeric_cols = ["frequency_of_use", "charge_amount"]

    st2 = StrataColumnTransformer(
        [
            ("strataencoder", StrataBuilderEncoder(), categorical_cols),
            (
                "stratadiscretizer",
                StrataBuilderDiscretizer(n_bins=10),
                numeric_cols,
            ),
        ]
    )

    X2, _, _, st_r = st2.fit_transform(X_pd, times, events)
    assert type(X2) == pd.DataFrame

    assert X2.shape[1] == (
        X.shape[1] - (len(categorical_cols + numeric_cols))
    )  # strata cols are removed
    assert len(st_r) == X.shape[0]
    assert len(st_r.shape) == 1
    assert st_r.dtype == np.int64

    X3, _, _, st_r = st2.fit_transform(X_pl, times, events)
    assert type(X3) == pl.DataFrame
    assert X3.shape[1] == (
        X.shape[1] - (len(categorical_cols + numeric_cols))
    )  # strata cols are removed
    assert len(st_r) == X.shape[0]
    assert len(st_r.shape) == 1
    assert st_r.dtype == np.int64
