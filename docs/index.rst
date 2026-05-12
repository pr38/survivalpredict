..
   survivalpredict documentation master file, created by
   sphinx-quickstart on Mon Apr 13 17:46:34 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

###############################
 survivalpredict documentation
###############################

Survivalpredict is a Python package for survival analysis with a
statistical learning lens. Survival analysis is a branch of statistics
that frames its modeling on the assumption that all observations are
being pushed towards failure as time moves forward. Each observation is
assumed to have either `failed` at a given point in time or has yet to
fail at the last known interval in time and is `censored`. Survival
curves are the estimated probability of failure at different intervals
of time. SurvivalPredict has a singular focus on `survival curves` and
being able to tune models using statistical learning methodology.

Survival analysis was originally developed in the context of clinical
trials; but has applications in customer churn, lead
conversion,financial defaults, operational failures, and elsewhere.

Core features of Survivalpredict are:

-  Vectorised code base, with some necessary parts written in numba.
   Resulting in performat code.

-  Scikit-learn enspired Api. All estimators return survival curves on
   `predict`. Survivalpredict also has tooling to interlope with
   Scikit-learn directly.

-  Ability to easily cross-validate estimators with Brier scores.

-  First-class support for left-censorship and model stratification.

Survivalpredict makes some assumptions:

-  The `times` array, the last known observed time of an individual
   before the event or censorship, is to be encoded as an integer. It is
   assumed that `time` starts with 1, and each interval of time is
   equally important. It is advised to engineer time to max out at a few
   thousand; large values in the `time` array can trigger expensive
   computation on several estimators.

-  The `events` array should be of boolean type, `True` if the
   individual experiences an event (e.g., charged, death, conversion,
   etc.), and `False` otherwise.

-  If the data is left-censored, in cases of time-varying effects or
   recurrent events, the smallest value of the `times_start` array
   should be 0, and the `times_start` array should be smaller than the
   `times` array.

-  When calling `predict` on an estimator, columns of the output will
   correspond to all times till max time, starting at time 1.

.. toctree::
   :maxdepth: 1
   :caption: Walkthroughs:

   General walkthrough<_collections/notebooks/survivalpredict_walkthrough_demo.ipynb>

   Interfacing with Sklearn<_collections/notebooks/demo_sklearn_interface.ipynb>

.. toctree::
   :maxdepth: 2
   :caption: API:

   api/datasets
   api/estimators
   api/metrics
   api/model_selection
   api/pipeline
   api/strata_preprocessing
   api/validation
