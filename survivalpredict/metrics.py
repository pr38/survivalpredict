from typing import Optional

import numpy as np

from .nonparametric import get_kaplan_meier_survival_curve_from_time_as_int_
from .utils import _as_bool_np_array, _as_int, _as_int_np_array, _as_numeric_np_array


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

    unique_times = np.arange(1, max_time + 1)

    kaplan_meier_survival_curve_inverted_event = (
        get_kaplan_meier_survival_curve_from_time_as_int_(
            1 - events_for_ipcw, times_for_ipcw, max_time
        )
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
    predictions: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    events_for_ipcw: Optional[np.ndarray] = None,
    times_for_ipcw: Optional[np.ndarray] = None,
    max_time: Optional[int] = None,
) -> np.ndarray:
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
    predictions: np.ndarray,
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
        integrated_brier_score = np.trapz(bs, unique_times) / unique_times[-1]

    else:
        integrated_brier_score = np.trapz(bs, unique_times)

    return integrated_brier_score


def integrated_brier_score_ipcw(
    predictions: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    events_for_ipcw: Optional[np.ndarray] = None,
    times_for_ipcw: Optional[np.ndarray] = None,
    average_by_time: Optional[bool] = False,
    max_time: Optional[int] = None,
) -> np.ndarray:
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


def _brier_scores_administrative(
    predictions: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    max_time: Optional[int] = None,
):
    if max_time is None:
        max_time = int(times.max())

    unique_times = np.arange(1, max_time + 1)

    survived_at_times = times[:, np.newaxis] > unique_times
    not_survived_at_times = ~survived_at_times
    not_survived_at_times_with_event = np.logical_and(
        not_survived_at_times, events[:, np.newaxis]
    )

    not_censored = np.logical_or(survived_at_times, events[:, np.newaxis])

    left = not_survived_at_times_with_event * np.square(predictions)
    right = survived_at_times * np.square(1 - predictions)

    scores_sumed = np.sum(left + right, axis=0)

    not_censored = np.logical_or(survived_at_times, events[:, np.newaxis])
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
    events: np.ndarray,
    times: np.ndarray,
    max_time: Optional[int] = None,
):
    times = _as_int_np_array(times)
    events = _as_bool_np_array(events)
    predictions = _as_numeric_np_array(predictions)

    if max_time is not None:
        max_time = _as_int(max_time, "max_time")

    return _brier_scores_administrative(predictions, events, times, max_time)


def _integrated_brier_score_administrative(
    predictions: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    max_time: Optional[int] = None,
    average_by_time: Optional[bool] = False,
):

    if max_time is None:
        max_time = int(times.max())

    unique_times = np.arange(1, max_time + 1)

    bs = _brier_scores_administrative(predictions, events, times, max_time=max_time)

    if average_by_time:
        integrated_brier_score = np.trapz(bs, unique_times) / unique_times[-1]

    else:
        integrated_brier_score = np.trapz(bs, unique_times)

    return integrated_brier_score


def integrated_brier_score_administrative(
    predictions: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    max_time: Optional[int] = None,
    average_by_time: Optional[bool] = False,
):
    times = _as_int_np_array(times)
    events = _as_bool_np_array(events)
    predictions = _as_numeric_np_array(predictions)

    if max_time is not None:
        max_time = _as_int(max_time, "max_time")

    if average_by_time is not None:
        average_by_time = _as_int(average_by_time, "average_by_time")

    return _integrated_brier_score_administrative(
        predictions, events, times, max_time, average_by_time
    )


__all__ = [
    "brier_scores_ipcw",
    "integrated_brier_score_ipcw",
    "brier_scores_administrative",
    "integrated_brier_score_administrative",
]
