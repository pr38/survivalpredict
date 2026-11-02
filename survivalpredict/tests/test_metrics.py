import numpy as np

from ..metrics import (
    brier_scores_administrative,
    integrated_brier_score_administrative,
    brier_scores_ipcw,
    integrated_brier_score_ipcw,
)

times_start = np.repeat(np.array([0, 1, 2, 3, 0, 1, 2, 3]), 3)
times = np.repeat(np.array([1, 2, 3, 4, 1, 2, 3, 10]), 3)
events = np.repeat(np.array([True, False, True, True, False, False, True, True]), 3)


def test_brier_scores():
    predict = np.random.random((times.shape[0], 10))

    for metric in [
        brier_scores_administrative,
        integrated_brier_score_administrative,
        brier_scores_ipcw,
        integrated_brier_score_ipcw,
    ]:
        metric(predict, times, events)


def test_brier_scores_with_left_censorship():
    predict = np.random.random((times.shape[0], 10))

    for metric in [brier_scores_administrative, integrated_brier_score_administrative]:
        metric(predict, times, events, times_start=times_start)
