from typing import Optional

import numpy as np
from sklearn.metrics import make_scorer

from ._data_validation import (
    _as_bool_np_array,
    _as_int,
    _as_int_np_array,
    _as_numeric_np_array,
)
from ._estimator_utils import _unpack_sklearn_pipeline_target
from ._nonparametric import get_kaplan_meier_survival_curve

__all__ = [
    "brier_scores_ipcw",
    "integrated_brier_score_ipcw",
    "integrated_brier_score_ipcw_sklearn_metric",
    "integrated_brier_score_ipcw_sklearn_scorer",
    "brier_scores_administrative",
    "integrated_brier_score_administrative",
    "integrated_brier_score_administrative_sklearn_metric",
    "integrated_brier_score_administrative_sklearn_scorer",
]


def _brier_scores_ipcw(
    predictions: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    events_for_ipcw: Optional[np.ndarray] = None,
    times_for_ipcw: Optional[np.ndarray] = None,
    max_time: Optional[int] = None,
):

    if (times_for_ipcw is None) or (events_for_ipcw is None):
        times_for_ipcw = times
        events_for_ipcw = events

    times_index = times - 1

    n_row = times.shape[0]

    if max_time is None:
        max_time = int(max([times_for_ipcw.max(), times.max()]))

    if predictions.shape[1] > max_time:
        predictions = predictions[:, max_time]

    unique_times = np.arange(1, max_time + 1)


    kaplan_meier_survival_curve_inverted_event = get_kaplan_meier_survival_curve(
        np.logical_not(events_for_ipcw), times_for_ipcw, max_time
    )

    survived_at_times = times[:, np.newaxis] > unique_times
    not_survived_at_times = ~survived_at_times
    not_survived_at_times_with_event = np.logical_and(
        not_survived_at_times, events[:, np.newaxis]
    )

    left_weights = np.divide(
        1,
        kaplan_meier_survival_curve_inverted_event[times_index][:, np.newaxis],
        out=np.zeros((n_row, 1)),
        where=kaplan_meier_survival_curve_inverted_event[times_index][:, np.newaxis]
        != 0,
    )
    right_weights = np.divide(
        1,
        kaplan_meier_survival_curve_inverted_event,
        out=np.zeros(max_time),
        where=kaplan_meier_survival_curve_inverted_event != 0,
    )

    left_weighted = (
        not_survived_at_times_with_event * np.square(predictions) * left_weights
    )
    right_weighted = survived_at_times * np.square(1 - predictions) * right_weights

    brier_scores_ipcw = np.mean(left_weighted + right_weighted, axis=0)

    return brier_scores_ipcw


def brier_scores_ipcw(
    predictions: np.ndarray[tuple[int,int],np.dtype[np.float64]],
    times: np.ndarray,
    events: np.ndarray,
    events_for_ipcw: Optional[np.ndarray] = None,
    times_for_ipcw: Optional[np.ndarray] = None,
    max_time: Optional[int] = None,
) -> np.ndarray:
    """
    Brier scores weighted with the inverse probability of censoring.

    Brier score for cases where censoring includes drop-up or loss of
    follow-up, a common occurrence in medical research. Each point in time
    within the survival curve is treated as an ‘outcome’ in the scores. The
    squared difference between an estimated probability is weighted using an
    inverse-event estimation. Weights are generated with the multiplicative
    inverse of an event-switched Kaplan-Meier estimator , where the Boolean
    values of events are inverted. For each interval of time of interest, if an
    observation is censored or expeiences event, those scores are weighted with
    the inverted Kaplan-Meier value for the last known time of that
    observation. If an individual is believed to be ‘alive’ at the point in
    time of interest, said scores are weighted with that time of interest's
    inverted Kaplan-Meier weights.

    Parameters
    ----------
    predictions : np.ndarray[tuple[int,int],np.dtype[np.float64]]
        Predicted survival curves.

    times : np.ndarray
        True points in time that were last observed.

    events : np.ndarray
        True indicators if event was experienced.

    events_for_ipcw : Optional[np.ndarray], default=None
        Events to build inverse probability of censoring weights on, it is
        acceptable to put in training events here. If None, will use Events.

    times_for_ipcw : Optional[np.ndarray], default=None
        Times to build inverse probability of censoring weights on, it is
        acceptable to put in training times here. If None, will use Times.

    max_time : Optional[int], default=None
        Maximum time to evaluate survival curves. If None, will evaluate all
        times seen.

    Returns
    -------
    np.ndarray
        Brier scores, starting from time 1 to max times.

    References
    ----------

    [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher, “Assessment and
    comparison of prognostic classification schemes for survival data,”
    Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    times = _as_int_np_array(times)
    events = _as_bool_np_array(events)
    predictions = _as_numeric_np_array(predictions)

    if events_for_ipcw is not None:
        events_for_ipcw = _as_bool_np_array(events_for_ipcw)

    if times_for_ipcw is not None:
        events_for_ipcw = _as_int_np_array(events_for_ipcw)

    if max_time is not None:
        max_time = _as_int(max_time, "max_time")

    return _brier_scores_ipcw(
        predictions, events, times, events_for_ipcw, times_for_ipcw, max_time
    )


def _integrated_brier_score_ipcw(
    predictions: np.ndarray[tuple[int,int],np.dtype[np.float64]],
    events: np.ndarray,
    times: np.ndarray,
    events_for_ipcw: Optional[np.ndarray] = None,
    times_for_ipcw: Optional[np.ndarray] = None,
    average_by_time: Optional[bool] = False,
    max_time: Optional[int] = None,
):

    if max_time is None:
        if (times_for_ipcw is None) or (events_for_ipcw is None):
            times_for_ipcw = times
            events_for_ipcw = events

        max_time = int(max([times_for_ipcw.max(), times.max()]))

    unique_times = np.arange(1, max_time + 1)

    bs = _brier_scores_ipcw(
        predictions,
        events,
        times,
        events_for_ipcw=events_for_ipcw,
        times_for_ipcw=times_for_ipcw,
        max_time=max_time,
    )

    if average_by_time:
        integrated_brier_score = np.trapezoid(bs, unique_times) / unique_times[-1]

    else:
        integrated_brier_score = np.trapezoid(bs, unique_times)

    return integrated_brier_score


def integrated_brier_score_ipcw(
    predictions: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    events_for_ipcw: Optional[np.ndarray] = None,
    times_for_ipcw: Optional[np.ndarray] = None,
    average_by_time: Optional[bool] = False,
    max_time: Optional[int] = None,
) -> np.ndarray:
    """
    Integrated Brier score weighted with the inverse probability of censoring.

    Integral of Brier scores with inverse probability of censoring, to allow
    for a singular metric of performance. Lower the better.

    Integral Brier scores for cases where censoring includes drop-up or loss of
    follow-up, a common occurrence in medical research. Each point in time
    within the survival curve is treated as an ‘outcome’ in the scores. The
    squared difference between an estimated probability is weighted using an
    inverse-event estimation. Weights are generated with the multiplicative
    inverse of an event-switched Kaplan-Meier estimator , where the Boolean
    values of events are inverted. For each interval of time of interest, if an
    observation is censored or expeiences event, those scores are weighted with
    the inverted Kaplan-Meier value for the last known time of that
    observation. If an individual is believed to be ‘alive’ at the point in
    time of interest, said scores are weighted with that time of interest's
    inverted Kaplan-Meier weights.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted survival curves.

    times : np.ndarray
        True points in time that were last observed.

    events : np.ndarray
        True indicators if event was experienced.

    events_for_ipcw : Optional[np.ndarray], default=None
        Events to build inverse probability of censoring weights on, it is
        acceptable to put in training events here. If None, will use Events.

    times_for_ipcw : Optional[np.ndarray], default=None
        Times to build inverse probability of censoring weights on, it is
        acceptable to put in training times here. If None, will use Times.

    average_by_time : Optional[bool], default=False
        If True, all scores are divided by the score of max time before getting
        the integral . If True, the integral stops being a metric to compare
        the performance of different models. This parameter exists to allow
        parroting with other packages.

    max_time : Optional[int], default=None
        Maximum time to evaluate survival curves. If None, will evaluate all
        times seen.

    Returns
    -------
    np.ndarray
        Integral of Brier scores.

    References
    ----------
    [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher, “Assessment and
    comparison of prognostic classification schemes for survival data,”
    Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    
    times = _as_int_np_array(times)
    events = _as_bool_np_array(events)
    predictions = _as_numeric_np_array(predictions)

    if events_for_ipcw is not None:
        events_for_ipcw = _as_bool_np_array(events_for_ipcw)

    if times_for_ipcw is not None:
        events_for_ipcw = _as_int_np_array(events_for_ipcw)

    if max_time is not None:
        max_time = _as_int(max_time, "max_time")

    if average_by_time is not None:
        average_by_time = _as_int(average_by_time, "max_time")

    return _integrated_brier_score_ipcw(
        predictions,
        events,
        times,
        events_for_ipcw,
        times_for_ipcw,
        average_by_time,
        max_time,
    )


def integrated_brier_score_ipcw_sklearn_metric(
    y_true: np.ndarray, y_pred: np.ndarray[tuple[int, int], np.dtype[np.floating]]
):
    """
    Scikit-learn metric wrapper for integrated Brier scores weighted with the
    inverse probability of censoring.

    Parameters
    ----------
    y_true : np.ndarray
        The true target; it is assumed, the target is generated from
        ‘survivalpredict.pipeline.build_sklearn_pipeline_target’.

    y_pred : np.ndarray[tuple[int, int], np.dtype[np.floating]]
        Predicted survival curves from a ‘survivalpredict’ estimator.
    """
    times, events, _, _ = _unpack_sklearn_pipeline_target(y_true)

    max_time = y_pred.shape[1]

    return integrated_brier_score_ipcw(y_pred, times, events, max_time=max_time)


integrated_brier_score_ipcw_sklearn_scorer = make_scorer(
    integrated_brier_score_ipcw_sklearn_metric, greater_is_better=False
)


def _brier_scores_administrative(
    predictions: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    max_time: Optional[int] = None,
    times_start: Optional[np.ndarray] = None,
):
    if max_time is None:
        max_time = int(times.max())

    if predictions.shape[1] > max_time:
        predictions = predictions[:, max_time]

    unique_times = np.arange(1, max_time + 1)

    survived_at_times = times[:, np.newaxis] > unique_times
    not_survived_at_times = ~survived_at_times
    not_survived_at_times_with_event = np.logical_and(
        not_survived_at_times, events[:, np.newaxis]
    )

    not_censored = np.logical_or(survived_at_times, events[:, np.newaxis])

    if times_start is not None:
        not_left_censored = ~unique_times <= times_start[:, np.newaxis]
        not_censored = np.logical_and(not_censored, not_left_censored)

    left = (not_survived_at_times_with_event * np.square(predictions)) * not_censored
    right = (survived_at_times * np.square(1 - predictions)) * not_censored

    scores_sumed = np.sum(left + right, axis=0)

    n_individuals_at_risk_at_time = not_censored.sum(axis=0)

    brier_scores = np.divide(
        scores_sumed,
        n_individuals_at_risk_at_time,
        out=np.zeros(max_time),
        where=n_individuals_at_risk_at_time != 0,
    )

    return brier_scores


def brier_scores_administrative(
    predictions: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    max_time: Optional[int] = None,
    times_start: Optional[np.ndarray] = None,
):
    """
    Brier scores for administrative censorship.

    Administrative brier scores is appropriate  in cases where censoring  is a
    function of time. When a individual is marked as censored due to the fact
    that they have not experainced event at the last observed time, and not due
    to a lack of follow-up or drop out. This is ideal for industry, where
    churned/converted/failed individuals are often known with relative
    certainty.

    Within survival analysis, each ‘outcome’ of the Brier scores is a distinct
    time within the survival curve. Administrative Brier scores simply look at
    the squared difference between the survival curve and the known survival;
    censored times for an individual are excluded from the score. The lower the
    scores, the better.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted survival curves.

    times : np.ndarray
        True points in time that were last observed.

    events : np.ndarray
        True indicators if event was experienced.

    max_time : Optional[int], default=None
        Maximum time to evaluate survival curves. If None, will evaluate all
        times seen.

    times_start : Optional[np.ndarray], default=None
        Starting point for observation. If not passed in, all times_start times
        are assumed to be 0.

    Returns
    -------
    ndarray of shape (n_samples, max_time), dtype=np.float64
        Brier scores, starting from time 1 to max times.

    References
    ----------
    [1] Kvamme, Håvard & Borgan, Ørnulf. (2019). The Brier Score under
    Administrative Censoring: Problems and Solutions.
    10.48550/arXiv.1912.08581.
    """
    times = _as_int_np_array(times)
    events = _as_bool_np_array(events)
    predictions = _as_numeric_np_array(predictions)

    if max_time is not None:
        max_time = _as_int(max_time, "max_time")

    return _brier_scores_administrative(
        predictions, events, times, max_time, times_start=times_start
    )


def _integrated_brier_score_administrative(
    predictions: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    max_time: Optional[int] = None,
    average_by_time: Optional[bool] = False,
    times_start: Optional[np.ndarray] = None,
):

    if max_time is None:
        max_time = int(times.max())

    unique_times = np.arange(1, max_time + 1)

    bs = _brier_scores_administrative(
        predictions, events, times, max_time=max_time, times_start=times_start
    )

    if average_by_time:
        integrated_brier_score = np.trapezoid(bs, unique_times) / unique_times[-1]

    else:
        integrated_brier_score = np.trapezoid(bs, unique_times)

    return integrated_brier_score


def integrated_brier_score_administrative(
    predictions: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    max_time: Optional[int] = None,
    average_by_time: Optional[bool] = False,
    times_start: Optional[np.ndarray] = None,
):
    """
    Integrated Brier scores for administrative censorship.

    Integral of Brier scores for administrative censorship, to allow for a
    singular metric of performance. Lower the better.

    Administrative brier scores is appropriate  in cases where censoring  is a
    function of time. When a individual is marked as censored due to the fact
    that they have not experainced event at the last observed time, and not due
    to a lack of follow-up or drop out. This is ideal for industry, where
    churned/converted/failed individuals are often known with relative
    certainty.

    Within survival analysis, each ‘outcome’ of the Brier scores is a distinct
    time within the survival curve. Administrative Brier scores simply look at
    the squared difference between the survival curve and the known survival;
    censored intervals are excluded from the score.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted survival curves.

    times : np.ndarray
        True points in time that were last observed.

    events : np.ndarray
        True indicators if event was experienced.

    max_time : Optional[int], default=None
        Maximum time to evaluate survival curves. If None, will evaluate all
        times seen.

    average_by_time : Optional[bool], default=False
        If True, all scores are divided by the score of max time before getting
        the integral . If True, the integral stops being a metric to compare
        the performance of different models. This parameter exists to allow
        parroting with other packages.

    times_start : Optional[np.ndarray], default=None
        Starting point for observation. If not passed in, all times_start times
        are assumed to be 0.

    Returns
    -------
    ndarray of shape (n_samples), dtype=np.float64
        Integral of Brier scores.

    References
    ----------
    [1] Kvamme, Håvard & Borgan, Ørnulf. (2019). The Brier Score under
    Administrative Censoring: Problems and Solutions.
    10.48550/arXiv.1912.08581.
    """

    times = _as_int_np_array(times)
    events = _as_bool_np_array(events)
    predictions = _as_numeric_np_array(predictions)

    if max_time is not None:
        max_time = _as_int(max_time, "max_time")

    if average_by_time is not None:
        average_by_time = _as_int(average_by_time, "average_by_time")

    return _integrated_brier_score_administrative(
        predictions,
        events,
        times,
        max_time,
        average_by_time,
        times_start=times_start,
    )


def integrated_brier_score_administrative_sklearn_metric(
    y_true: np.ndarray, y_pred: np.ndarray[tuple[int, int], np.dtype[np.floating]]
):
    """
    Scikit-learn metric for Integral of brier scores for administrative
    censorship.

    Parameters
    ----------
    y_true : np.ndarray
        The true target; it is assumed, the target is generated from
        ‘survivalpredict.pipeline.build_sklearn_pipeline_target’.

    y_pred : np.ndarray[tuple[int, int], np.dtype[np.floating]]
        Predicted survival curves from a ‘survivalpredict’ estimator.
    """

    times, events, _, times_start = _unpack_sklearn_pipeline_target(y_true)

    max_time = y_pred.shape[1]

    return integrated_brier_score_administrative(
        y_pred, times, events, max_time=max_time, times_start=times_start
    )


integrated_brier_score_administrative_sklearn_scorer = make_scorer(
    integrated_brier_score_administrative_sklearn_metric, greater_is_better=False
)
