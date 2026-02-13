# SurvivalPredict

A python packaged centered around Survival Analysis Statistical Learning, meaning the prediction of survival acoss time.

WIP. A pypi release should be released soon. In the meantime, the code in this repo can be installed via `pip install git+https://github.com/pr38/survivalpredict`. Ideally, before the first pypi release, some left-censoring support, docstrings, example notebooks, as well as implementations for Aalen’s additive hazards' and 'multi-task logistic regression' models should be added. With the goal of adding sparse data support as well as  tree-based, ensemble, and exotic neural network models further down the line. 



## Estimators
Below are the estimators implemented in the `survivalpredict.estimators` sub-module

<table>
    <tr>
        <th>Estimators</th>
        <th>Description</th>
        <th>Stratifiable</th>
    </tr>
    <tr>
        <td>CoxProportionalHazard</td>
        <td>
        Cox proportional hazards model is a linear semi-parametric relative risk model. A staple of survival analysis. Fast and efficient to train. Survivalpredict's implementation has many optimizations and is up to 10x to 20x faster than other implementations available to Python. Both breslow and efron ties are supported. Currently only the breslow base hazard is avalable.
        </td>
        <th>Yes</th>
    </tr>
    <tr>
        <td>ParametricDiscreteTimePH</td>
        <td> A fully parametric linear hazards model. Chen, weibull, log_normal, log_logistic, gompertz and additive_chen_weibull baseline hazards are available as hyperparameters. Maximum likelihood is estimated using a survival distinct time likelihood with censorship. Implemented with Pymc/Pytensor, with either a Jax or numba backend.</td>
        <th>Yes</th>
    </tr>
    <tr>
        <td>KaplanMeierSurvivalEstimator</td>
        <td> Univariate non-parametric survival curve. Useful as a baseline/dummy estimator.</td>
        <th>Accepts strata, but builds a survival curve for each strata. </th>
    </tr>
        <tr>
        <td>KNeighborsSurvival</td>
        <td>K nearest neighbors for survival. An in-memory non-parametric model that builds a Kaplan-Meier survival curve based on neighbors.
        </td>
        <th>No </th>
    </tr>
        <td>CoxNNetPH</td>
        <td> A neural network model for estimating relative risk. Cox proportional hazards model's 'negative log likelihood for Breslow ties' is used as a loss function. Breslow's base hazard for relative risk is used to estimate survival across time. Implemented using Jax.  </td>
        <th>Yes</th>
</table>



## Metrics

Survivalpredict focuses on metrics that directly measure prediction performance. Hence, the `survivalpredict.metrics` module intentionally excludes metrics based on ranking relative risk(i.e., ' c-index').


<table>
    <tr>
        <th>Metrics</th>
        <th>Description</th>
    </tr>
    <tr>
        <th>brier_scores_administrative</th>
        <th>Squared error between the true survival and prediction for each time of interest. Censored intervals are ignored. Averaged by the number of rows not censored at a given interval of time. Ideal in cases of 'administrative' censorship, where 'survival time' is modeled after the time of an individual in the experiment, and not calendar time. This mertic is ideal for cases of churn, conversion and operational failure. See <a href=https://jmlr.org/papers/volume24/19-1030/19-1030.pdf>here</a></th>.
    </tr>
    <tr>
        <th>integrated_brier_score_administrative</th>
        <th>Integral of administrative brier scores, to allow for a singular metric of performance. </th>
    </tr>
        <tr>
        <th>integrated_brier_score_administrative_sklearn_metric</th>
        <th>scikit-learn metric wraper around `integrated_brier_score_administrative` function, for acessing said metric in when using the SklearnSurvivalPipeline wrapper class when interfacing with scikit-learn.</th>
    </tr>
    <tr>
        <th>integrated_brier_score_administrative_sklearn_scorer</th>
        <th>scikit-learn scorer wraper around `integrated_brier_score_administrative` function, for acessing said metric in when using the SklearnSurvivalPipeline wrapper class when interfacing with scikit-learn.</th>
    </tr>
    <tr>
        <th>brier_scores_ipcw</th>
        <th>Brier scores with inverse probability of censoring weights. The squared error between the true survival and prediction is weighted using a Kaplan-Meier curve with inverted events, depending on censoring and failure at different points in time. This is a common metric within the field of biostatistics and is used in clinical trials.See <a href=https://pubmed.ncbi.nlm.nih.gov/10474158>here</a></th>.</th>
    </tr>
    <tr>
        <th>integrated_brier_score_ipcw</th>
        <th>Integral of brier scores with probability of censoring weights, to allow for a singular metric of performance.</th>
    </tr>
    <tr>
        <th>integrated_brier_score_ipcw_sklearn_metric</th>
        <th>scikit-learn metric wraper around `integrated_brier_score_ipcw` function.</th>
    </tr>
        <tr>
        <th>integrated_brier_score_ipcw_sklearn_scorer</th>
        <th>scikit-learn scorer wraper around `integrated_brier_score_ipcw` function.</th>
    </tr>

</table>


## Strata Preprocessing

The `survivalpredict.strata_preprocessing` module allows for the creation of strata to be used various estimators.


<table>
    <tr>
        <th>Class</th>
        <th>Description</th>
    </tr>
        <tr>
        <th>StrataBuilderDiscretizer</th>
        <th>Builds strata keys from numeric data. Allows various splitting strategies. </th>
    </tr>
        <tr>
        <th>StrataBuilderEncoder</th>
        <th>Builds strata keys from categorical data.</th>
    </tr>
        <tr>
        <th>StrataColumnTransformer</th>
        <th>Allows various StrataBuilders to be stacked and simultaneously to be run on different columns to build the strata. Modeled after scikit-learn's ColumnTransformer. </th>
    </tr>
        <tr>
        <th>make_strata_column_transformer</th>
        <th>Generates the StrataColumnTransformer class without having to name each transformation directly, like scikit-learn's make_column_transformer. </th>
    </tr>
</table>



## Pipeline

Due to various reasons, survivalpredict intentionaly breaks with scikit-learn's api in several ways. The `survivalpredict.pipeline` module allows for creating wrappers around various survivalpredict classes, in order for survivalpredict intperpolate with the greater scikit-learn ecosysteam(ie, for feature selection or hyperparameter tuning); in addition of the various utility of a conventional scikit-learn's pipeline. 

<table>
    <tr>
        <th>Class</th>
        <th>Description</th>
    </tr>
    <tr>
        <th>build_sklearn_pipeline_target</th>
        <th>Builds a singular target array from the times and events arrays. Used as the 'y'/observed for scikit-learn ecosystem.</th>
    </tr>
    <tr>
        <th>SklearnSurvivalPipeline</th>
        <th>Stacks various sklearn transformers and survivalpredict strata_builders and estimators into single class. It assumes the output of the `build_sklearn_pipeline_target` function as the 'y'/observed.</th>
    </tr>
    <tr>
        <th>make_sklearn_survival_pipeline</th>
        <th>Generates a SklearnSurvivalPipeline class without having to directly name all the steps.</th>
    </tr>
</table>

## Validation

survivalpredict comes with some native model validation capability, within `survivalpredict.validation`.

<table>
    <tr>
        <th>Class</th>
        <th>Description</th>
    </tr>
        <tr>
        <th>sur_cross_val_score</th>
        <th>survivalpredict's equivalent to scikit-learn's cross_val_score.</th>
    </tr>
        <tr>
        <th>sur_cross_validate</th>
        <th>survivalpredict's equivalent to scikit-learn's cross_validate.</th>
    </tr>
</table>


## Model Selection

Scikit-learn's model_selection is also mimicked within `survivalpredict.model_selection`

<table>
    <tr>
        <th>Class</th>
        <th>Description</th>
    </tr>
        <tr>
        <th>Sur_GridSearchCV</th>
        <th>survivalpredict's equivalent to scikit-learn's GridSearchCV</th>
    </tr>
        <tr>
        <th>Sur_RandomizedSearchCV</th>
        <th>survivalpredict's equivalent to scikit-learn's RandomizedSearchCV</th>
    </tr>
</table>



