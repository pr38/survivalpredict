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
)


@nb.njit(_estimate_allen_additive_hazard_time_weights_sig,cache=True)
def _estimate_allen_additive_hazard_time_weights(X, times, events):
    hazard_weights_times = np.unique(times[events])
    hazard_weights = np.empty((hazard_weights_times.shape[0], X.shape[1]))

    for i, t in enumerate(hazard_weights_times):
        survived_at_time = times >= t
        exits = times == t
        deaths_mask = np.logical_and(exits, events)
        deaths_at_times = np.logical_and(survived_at_time, deaths_mask)

        if survived_at_time.any() and deaths_at_times.sum() > 1:
            death_as_target = deaths_mask[survived_at_time].astype(np.float64)

            X_mask = X[survived_at_time, :]

            a = np.dot(X_mask.T, X_mask)
            b = np.dot(X_mask.T, death_as_target)

            w = np.linalg.solve(a, b)
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


@nb.njit(_generate_hazards_at_times_from_allen_additive_hazard_weights_sig,cache=True)
def _generate_hazards_at_times_from_allen_additive_hazard_weights(
    X, hazard_weights, hazard_weights_times, max_time
):
    hazards = np.zeros((X.shape[0], max_time))

    for t, w in zip(hazard_weights_times, hazard_weights):
        hazards[:, t] = np.dot(X, w)

    return hazards
