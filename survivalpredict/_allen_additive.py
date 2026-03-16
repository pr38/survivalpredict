import numba as nb
import numpy as np

_estimate_allen_additive_hazard_time_weights_sig = nb.types.Tuple(
    (
        nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    )
)(
    nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.bool_, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.float64,
)


@nb.njit(_estimate_allen_additive_hazard_time_weights_sig, cache=True)
def _estimate_allen_additive_hazard_time_weights(X, times, events, times_start, alpha):
    hazard_weights_times = np.unique(times[events])
    hazard_weights = np.empty((hazard_weights_times.shape[0], X.shape[1]))

    for i, t in enumerate(hazard_weights_times):
        survived_at_time = times >= t
        exits = times == t
        deaths_mask = np.logical_and(exits, events)
        not_right_censored = np.logical_and(survived_at_time, deaths_mask)
        not_left_censored = times_start < t
        not_censored = np.logical_or(not_right_censored, not_left_censored)

        if not_censored.any() and not_censored.sum() > 1:
            death_as_target = deaths_mask[not_censored].astype(np.float64)

            X_mask = X[not_censored, :]

            a = np.dot(X_mask.T, X_mask) + alpha
            b = np.dot(X_mask.T, death_as_target)

            try:
                w = np.linalg.solve(a, b)
            except:
                w = np.zeros(X.shape[1])
   
            hazard_weights[i, :] = w

    return hazard_weights, hazard_weights_times


_generate_hazards_at_times_from_allen_additive_hazard_weights_sig = nb.types.Array(
    nb.types.float64, 2, "C", False, aligned=True
)(
    nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
    nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.int64,
)


@nb.njit(_generate_hazards_at_times_from_allen_additive_hazard_weights_sig, cache=True)
def _generate_hazards_at_times_from_allen_additive_hazard_weights(
    X, hazard_weights, hazard_weights_times, max_time
):
    hazards = np.zeros((X.shape[0], max_time))

    for t, w in zip(hazard_weights_times, hazard_weights):
        hazards[:, t] = np.dot(X, w)

    return hazards
