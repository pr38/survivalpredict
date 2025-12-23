import numba as nb
import numpy as np

from ._nonparametric import get_kaplan_meier_survival_curve_from_time_as_int_

build_kaplan_meier_survival_curve_from_neighbors_indexes_siganture = nb.types.Array(
    nb.types.float64, 2, "C", False, aligned=True
)(
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.boolean, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 2, "C", False, aligned=True),
    nb.types.int64,
)


@nb.njit(
    build_kaplan_meier_survival_curve_from_neighbors_indexes_siganture,
    cache=True,
    parallel=True,
)
def build_kaplan_meier_survival_curve_from_neighbors_indexes(
    times, event, neighbors_indexes, max_time
):

    predictions = np.empty((neighbors_indexes.shape[0], max_time))

    n_rows = neighbors_indexes.shape[0]

    for i in nb.prange(n_rows):
        neighbors_index = neighbors_indexes[i]
        neighbors_events = event[neighbors_index]
        neighbors_times = times[neighbors_index]
        neighbors_kaplan_meier = get_kaplan_meier_survival_curve_from_time_as_int_(
            neighbors_events, neighbors_times, max_time
        )
        predictions[i] = neighbors_kaplan_meier

    return predictions
