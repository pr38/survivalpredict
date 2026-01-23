import itertools

import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score

from ..datasets import load_iranian_telecom_churn
from ..estimators import CoxProportionalHazard
from ..metrics import integrated_brier_score_administrative_sklearn_scorer
from ..pipeline import (
    SklearnSurvivalPipeline,
    build_sklearn_pipeline_target,
    make_sklearn_survival_pipeline,
)
from ..strata_preprocessing import (
    StrataBuilderEncoder,
    StrataColumnTransformer,
    make_strata_column_transformer,
)


@pytest.fixture
def X_df_y():
    iranian_telecom_churn = load_iranian_telecom_churn()

    X = iranian_telecom_churn["X"]
    times = iranian_telecom_churn["times"]
    events = iranian_telecom_churn["events"]
    column_names = iranian_telecom_churn["column_names"]

    X_df = pd.DataFrame(X, columns=column_names)

    y = build_sklearn_pipeline_target(times, events)

    return X_df, y


def test_pipeline_params():
    strata_column_transformer = make_strata_column_transformer(
        (StrataBuilderEncoder(), "tariff_plan")
    )
    pipe = make_sklearn_survival_pipeline(
        strata_column_transformer, CoxProportionalHazard(alpha=10), max_time=100
    )

    params = pipe.get_params()

    assert "stratacolumntransformer__stratabuilderencoder__columns" in params
    assert "coxproportionalhazard__alpha" in params
    assert (
        params["stratacolumntransformer__stratabuilderencoder__columns"]
        == "tariff_plan"
    )

    strata_column_transformer.set_params(
        **{"stratabuilderencoder__columns": "complains"}
    )

    params2 = pipe.get_params()

    assert (
        params2["stratacolumntransformer__stratabuilderencoder__columns"] == "complains"
    )


def test_pipeline_fit_predict(X_df_y):
    X_df, y = X_df_y

    max_time = int(np.percentile(y["times"], 75))

    strata_column_transformer = StrataColumnTransformer(
        [("stratabuilderencoder", StrataBuilderEncoder(), "complains")]
    )
    pipe = SklearnSurvivalPipeline(
        [
            ("stratacolumntransformer", strata_column_transformer),
            ("cox", CoxProportionalHazard(alpha=10)),
        ],
        max_time=max_time,
    )

    pipe.fit(X_df, y)

    r = pipe.predict(X_df)

    assert r.shape[0] == X_df.shape[0]
    assert r.shape[1] == max_time

    r = pipe.fit_predict(X_df, y)

    assert r.shape[0] == X_df.shape[0]
    assert r.shape[1] == max_time


def test_pipeline_on_sk_cross_val_score(X_df_y):
    X_df, y = X_df_y

    max_time = int(np.percentile(y["times"], 75))

    strata_column_transformer = StrataColumnTransformer(
        [("stratabuilderencoder", StrataBuilderEncoder(), "complains")]
    )
    pipe = SklearnSurvivalPipeline(
        [
            ("stratacolumntransformer", strata_column_transformer),
            ("coxproportionalhazard", CoxProportionalHazard(alpha=10)),
        ],
        max_time=max_time,
    )

    results = cross_val_score(
        pipe,
        X_df,
        y,
        scoring=integrated_brier_score_administrative_sklearn_scorer,
        cv=ShuffleSplit(10),
    )


def test_pipeline_on_sk_GridSearchCV(X_df_y):
    X_df, y = X_df_y

    max_time = int(np.percentile(y["times"], 75))

    strata_column_transformer = make_strata_column_transformer(
        (StrataBuilderEncoder(), "tariff_plan")
    )
    pipe = make_sklearn_survival_pipeline(
        strata_column_transformer, CoxProportionalHazard(alpha=10), max_time=max_time
    )

    cat_col_combonations = list(
        itertools.chain.from_iterable(
            itertools.combinations(["tariff_plan", "complains", "status"], i)
            for i in range(1, 2)
        )
    )

    pram_grid = {
        "stratacolumntransformer__stratabuilderencoder__columns": cat_col_combonations,
        "coxproportionalhazard__alpha": [0, 10, 100],
        "coxproportionalhazard__l1_ratio": [0.0, 0.25, 0.5, 0.75, 0.1],
    }

    gs = GridSearchCV(
        pipe,
        pram_grid,
        scoring=integrated_brier_score_administrative_sklearn_scorer,
        cv=13,
    )

    gs.fit(X_df, y)
