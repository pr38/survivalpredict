# SurvivalPredict

A python package centered around Survival Analysis Statistical Learning, for predicting survival curves. The code in this repo is lovingly written without any stochastic generative processes.

See <a href=https://survivalpredict.readthedocs.io/en/latest/index.html>documentation</a>

WIP. A pypi release should be released soon. In the meantime, the code in this repo can be installed via `pip install git+https://github.com/pr38/survivalpredict`. Ideally, before the first pypi release docstrings and example notebooks will be added. With the goal of finishing left-censoring support, sparse data support as well as  tree-based, ensemble, and exotic neural network  models further down the line. 


<a href=https://htmlpreview.github.io/?https://github.com/pr38/survivalpredict/blob/main/notebooks/survivalpredict_walkthrough_demo.html>General walkthrough-demo</a>

<a href=https://htmlpreview.github.io/?https://github.com/pr38/survivalpredict/blob/main/notebooks/demo_sklearn_interface.html>Demo for interfacing with scikit learn</a>



survivalpredict makes some assumptions. 
* The `times` array, the last known observed time of an individual before the event or censorship, is to be encoded as an integer. It is assumed that `time` starts with 1, and each interval of time is equally important. It is advised to engineer time to max out at a few thousand; large values in the `time` array can trigger expensive compute on several estimators.
*  The `events` array should be of boolean type, `True` if the individual experiences an event (e.g., charged, death, conversion, etc.), and `False` otherwise.
*  If the data is left-censored, in cases of time-varying effects or recurrent events, the smallest value of the `times_start` array should be 0, and the `times_start` array should be smaller than the `times` array.
*  When calling `predict` on an estimator, columns of the output will correspond to all times till max time, starting at time 1.


## Estimators
The estimators implemented in the `survivalpredict.estimators` sub-module.

<table>
    <tr>
        <th>Estimators</th>
        <th>Description</th>
        <th>Stratifiable</th>
        <th>Left-censorable</th>
    </tr>
    <tr>
        <td>CoxProportionalHazard</td>
        <td>
        Cox Proportional Hazards model is a linear semi-parametric relative risk model. A staple of survival analysis. Fast and efficient to train. Survivalpredict's implementation has many optimizations and is up to 10x to 20x faster than other implementations available to Python. Both breslow and efron ties are supported. Currently, only the Breslow base hazard is available.
        </td>
        <td> Yes</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>ParametricDiscreteTimePH</td>
        <td> A fully parametric linear hazards model. Chen, weibull, log_normal, log_logistic, gompertz, gamma and additive_chen_weibull baseline hazards are available as hyperparameters. Maximum likelihood is estimated using a survival distinct time likelihood with censorship. Implemented with Pymc/Pytensor, with either a Jax or numba backend.</td>
        <td>Yes</td>
        <td>Yes</td>
    </tr>
        <tr>
        <td>CoxElasticNetPH</td>
        <td>Cox Proportional Hazards model model with Elastic-Net/Lasso penalty and feature shrinkage/selection. Uses the 'Newton Raphson-like' coordinate descent algorithm described in <a href=https://pmc.ncbi.nlm.nih.gov/articles/PMC4824408>Simon, Noah et al. “Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent</a>. Assumes breslow ties. The current literature is unclear on how to incorporate stratification support into said algorithm.
        </td>
        <td> No</td>
        <td>Yes</td>
    </tr>
    <tr>
        <td>KaplanMeierSurvivalEstimator</td>
        <td> Univariate non-parametric survival curve. Useful as a baseline/dummy estimator.</td>
        <td>Accepts strata, but builds a survival curve for each strata. </td>
        <td>Yes</td>
    </tr>
        <tr>
        <td>KNeighborsSurvival</td>
        <td>K nearest neighbors for survival. An in-memory non-parametric model that builds a Kaplan-Meier survival curve based on neighbors.
        </td>
        <td>No</td>
        <td>Yes</td>
    </tr>
        <td>CoxNNetPH</td>
        <td> A neural network model for estimating relative risk. Cox proportional hazards model's 'negative log likelihood for Breslow ties' is used as a loss function. Breslow's base hazard for relative risk is used to estimate survival across time. Implemented using Jax.  </td>
        <td>Yes</td>
        <td>Yes</td>
    </tr>
        <tr>
        <td>AalenAdditiveHazard</td>
        <td> Linear multivariate non-parametric estimation of hazard. Allows for each interval of time and feature to have an associated coefficient, allowing for the effects of features to change over time.
        </td>
        <td>No</td>
        <td>Yes</td>
</table>



## Metrics

Survivalpredict focuses on metrics that directly measure prediction performance. Hence, the `survivalpredict.metrics` module intentionally excludes metrics based on ranking relative risk(i.e., ' c-index').

<table>
    <tr>
        <th>Metrics</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>brier_scores_administrative</td>
        <td>Squared error between the true survival and prediction for each time of interest. Censored intervals are ignored. Averaged by the number of rows not censored at a given interval of time. Ideal in cases of 'administrative' censorship, where 'survival time' is modeled after the time of an individual in the experiment, and not calendar time. This metric is ideal for cases of churn, conversion and operational failure. See <a href=https://jmlr.org/papers/volume24/19-1030/19-1030.pdf>here</a>.</td>
    </tr>
    <tr>
        <td>integrated_brier_score_administrative</td>
        <td>Integral of administrative brier scores, to allow for a singular metric of performance. </td>
    </tr>
        <td>integrated_brier_score_administrative_sklearn_metric</td>
        <td>scikit-learn metric wrapper around `integrated_brier_score_administrative` function, for accessing said metric in when using the SklearnSurvivalPipeline wrapper class when interfacing with scikit-learn.</td>
    </tr>
    <tr>
        <td>integrated_brier_score_administrative_sklearn_scorer</td>
        <td>scikit-learn scorer wrapper around `integrated_brier_score_administrative` function, for accessing said metric in when using the SklearnSurvivalPipeline wrapper class when interfacing with scikit-learn.</td>
    </tr>
    <tr>
        <td>brier_scores_ipcw</td>
        <td>Brier scores with inverse probability of censoring weights. The squared error between the true survival and prediction is weighted using a Kaplan-Meier curve with inverted events, depending on censoring and failure at different points in time. This is a common metric within the field of biostatistics and is used in clinical trials. See <a href=https://pubmed.ncbi.nlm.nih.gov/10474158>here</a>.</td>
    </tr>
    <tr>
        <td>integrated_brier_score_ipcw</td>
        <td>Integral of brier scores with probability of censoring weights, to allow for a singular metric of performance.</td>
    </tr>
    <tr>
        <td>integrated_brier_score_ipcw_sklearn_metric</td>
        <td>scikit-learn metric wrapper around `integrated_brier_score_ipcw` function.</td>
    </tr>
        <td>integrated_brier_score_ipcw_sklearn_scorer</td>
        <td>scikit-learn scorer wrapper around `integrated_brier_score_ipcw` function.</td>
    </tr>
</table>


## Strata Preprocessing

The `survivalpredict.strata_preprocessing` module allows for the creation of strata to be used various estimators.


<table>
    <tr>
        <th>Class</th>
        <th>Description</th>
    </tr>
        <td>StrataBuilderDiscretizer</td>
        <td>Builds strata keys from numeric data. Allows various splitting strategies. </td>
    </tr>
        <td>StrataBuilderEncoder</td>
        <td>Builds strata keys from categorical data.</td>
    </tr>
        <td>StrataColumnTransformer</td>
        <td>Allows various StrataBuilders to be stacked and simultaneously to be run on different columns to build the strata. Modeled after scikit-learn's ColumnTransformer. </td>
    </tr>
        <td>make_strata_column_transformer</td>
        <td>Generates the StrataColumnTransformer class without having to name each transformation directly, like scikit-learn's make_column_transformer. </td>
    </tr>
</table>



## Pipeline

Due to various reasons, survivalpredict intentionally breaks with scikit-learn's api in several ways. The `survivalpredict.pipeline` module allows for creating wrappers around various survivalpredict classes, in order for survivalpredict to interpolate with the greater scikit-learn ecosystem (i.e., for feature selection or hyperparameter tuning); in addition of the various utility of a conventional scikit-learn's pipeline. 

<table>
    <tr>
        <th>Class</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>build_sklearn_pipeline_target</td>
        <td>Builds a singular target array from the times and events arrays. Used as the 'y'/observed for scikit-learn ecosystem.</td>
    </tr>
    <tr>
        <td>SklearnSurvivalPipeline</td>
        <td>Stacks various sklearn transformers and survivalpredict strata_builders and estimators into single class. It assumes the output of the `build_sklearn_pipeline_target` function as the 'y'/observed.</td>
    </tr>
    <tr>
        <td>make_sklearn_survival_pipeline</td>
        <td>Generates a SklearnSurvivalPipeline class without having to directly name all the steps.</td>
    </tr>
</table>

## Validation

survivalpredict comes with some native model validation capability, within `survivalpredict.validation`.

<table>
    <tr>
        <th>Class</th>
        <th>Description</th>
    </tr>
        <td>sur_cross_val_score</td>
        <td>survivalpredict's equivalent to scikit-learn's cross_val_score.</td>
    </tr>
        <td>sur_cross_validate</td>
        <td>survivalpredict's equivalent to scikit-learn's cross_validate.</td>
    </tr>
</table>


## Model Selection

Scikit-learn's model_selection is also mimicked within `survivalpredict.model_selection`

<table>
    <tr>
        <th>Class</th>
        <th>Description</th>
    </tr>
        <td>Sur_GridSearchCV</td>
        <td>survivalpredict's equivalent to scikit-learn's GridSearchCV</td>
    </tr>
        <td>Sur_RandomizedSearchCV</td>
        <td>survivalpredict's equivalent to scikit-learn's RandomizedSearchCV</td>
    </tr>
</table>



