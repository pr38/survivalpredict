import numba as nb
import numpy as np

_unique_with_return_inverse_siganture = nb.types.Tuple(
    (
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    )
)(nb.types.Array(nb.types.int64, 1, "C", False, aligned=True))


@nb.njit(_unique_with_return_inverse_siganture, cache=True)
def _unique_with_return_inverse(ar):
    "equivalent of np.unique(x,return_inverse=True) for the nopython numba runtime"
    perm = ar.argsort(kind="mergesort")
    aux = ar[perm]

    mask = np.empty(aux.shape, dtype=np.bool)
    mask[:1] = True

    mask[1:] = aux[1:] != aux[:-1]

    imask = np.cumsum(mask) - 1
    inv_idx = np.empty(mask.shape, dtype=np.intp)
    inv_idx[perm] = imask

    return aux[mask], inv_idx


split_and_preprocess_data_by_strata_siganture = nb.types.Tuple(
    (
        nb.types.int64,
        nb.types.Array(nb.types.int64, 1, "C"),
        nb.types.List(nb.types.Array(nb.types.bool_, 1, "C")),
        nb.types.List(nb.types.Array(nb.types.int64, 1, "C")),
        nb.types.List(nb.types.Array(nb.types.float64, 2, "C")),
        nb.types.List(nb.types.Array(nb.types.int64, 1, "C")),
        nb.types.List(nb.types.Array(nb.types.float64, 1, "C")),
        nb.types.List(nb.types.int64),
    )
)(
    nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.bool_, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
)


@nb.njit(split_and_preprocess_data_by_strata_siganture, cache=True)
def split_and_preprocess_data_by_strata(X, times, events, strata):

    seen_strata, stata_index = _unique_with_return_inverse(strata)
    n_strata = len(seen_strata)

    strata_masks = [s == stata_index for s in range(n_strata)]

    events_strata = []
    X_strata = []
    times_strata = []
    time_return_inverse_strata = []
    n_unique_times_strata = []
    event_counts_at_times_strata = []

    for mask in strata_masks:
        X_s = X[mask]
        X_strata.append(X_s)
        events_s = events[mask]
        events_strata.append(events_s)

        times_s = times[mask]
        times_strata.append(times_s)

        unique_times_s, time_return_inverse_s = _unique_with_return_inverse(times_s)
        time_return_inverse_strata.append(time_return_inverse_s)

        n_unique_times_s = len(unique_times_s)
        n_unique_times_strata.append(n_unique_times_s)

        event_counts_at_times_s = np.bincount(time_return_inverse_s, weights=events_s)

        event_counts_at_times_strata.append(event_counts_at_times_s)

    return (
        n_strata,
        seen_strata,
        events_strata,
        times_strata,
        X_strata,
        time_return_inverse_strata,
        event_counts_at_times_strata,
        n_unique_times_strata,
    )


def preprocess_data_for_cox_ph(X, times, events, strata=None):
    if strata is not None:
        (
            n_strata,
            seen_strata,
            events_strata,
            times_strata,
            X_strata,
            time_return_inverse_strata,
            event_counts_at_times_strata,
            n_unique_times_strata,
        ) = split_and_preprocess_data_by_strata(X, times, events, strata)

    else:
        unique_times, time_return_inverse = np.unique(times, return_inverse=True)
        n_unique_times = len(unique_times)
        event_counts_at_times = np.bincount(
            time_return_inverse, weights=events.astype(np.int64)
        )
        n_strata = 0
        seen_strata = None
        X_strata = [X]
        events_strata = [events]
        time_return_inverse_strata = [time_return_inverse]
        n_unique_times_strata = [n_unique_times]
        event_counts_at_times_strata = [event_counts_at_times]
        n_strata = 1
        times_strata = [times]

    return (
        n_strata,
        seen_strata,
        X_strata,
        times_strata,
        events_strata,
        time_return_inverse_strata,
        n_unique_times_strata,
        event_counts_at_times_strata,
    )


index_per_not_censored_times_nb_signature = nb.types.Array(nb.types.int64, 1, "C")(
    nb.types.Array(nb.types.int64, 1, "C"), nb.types.Array(nb.types.bool_, 1, "C")
)


@nb.njit(index_per_not_censored_times_nb_signature, cache=True)
def get_index_per_not_censored_times(time, events):
    last_time = np.inf
    current_index = 0
    indexes = []

    for i in range(time.shape[0]):
        t = time[i]
        e = events[i]

        not_censored = e == 1

        if t == last_time and not_censored:
            current_index = current_index + 1

        elif not_censored:
            current_index = 0
        else:
            current_index = -1

        last_time = t

        indexes.append(current_index)

    return np.array(indexes)


get_l_div_m_stata_per_strata_signature = nb.types.List(
    nb.types.Array(nb.types.float64, 1, "C")
)(
    nb.types.List(nb.types.Array(nb.types.bool_, 1, "C"), True),
    nb.types.List(nb.types.Array(nb.types.int64, 1, "C"), True),
    nb.types.List(nb.types.Array(nb.types.int64, 1, "C"), True),
    nb.types.List(nb.types.int64, True),
)


@nb.njit(get_l_div_m_stata_per_strata_signature, cache=True)
def get_l_div_m_stata_per_strata(
    events_strata, times_strata, time_return_inverse_strata, n_unique_times_strata
):

    l_div_m_stata = []

    for times, time_return_inverse, n_unique_times, events in zip(
        times_strata, time_return_inverse_strata, n_unique_times_strata, events_strata
    ):

        death_per_time = np.bincount(
            time_return_inverse, weights=events, minlength=n_unique_times
        )[time_return_inverse]

        index_per_not_censored_times = get_index_per_not_censored_times(times, events)

        n_rows = len(events)

        l_div_m_ = np.zeros(n_rows)

        for i in range(n_rows):
            e_i = events[i]
            d_i = death_per_time[i]
            if d_i != 0 and e_i == 1:
                i_i = index_per_not_censored_times[i]
                l_div_m_[i] = i_i / d_i

        l_div_m_stata.append(l_div_m_)

    return l_div_m_stata


map_new_strata_signature = nb.types.Tuple(
    (nb.types.Array(nb.types.int64, 1, "C", False, aligned=True), nb.types.boolean)
)(
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
    nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
)


@nb.jit(map_new_strata_signature, cache=True)
def map_new_strata(strata, seen_strata):
    has_unseen_strata = False
    n_rows = len(strata)
    n_strata = len(seen_strata)
    n_strata_plus_1 = n_strata + 1
    output = np.repeat(-1, n_rows)
    for ri in range(n_rows):
        strata_ri = strata[ri]
        for s_i in range(n_strata_plus_1):
            if s_i != n_strata and strata_ri == seen_strata[s_i]:
                output[ri] = s_i
                break
            elif s_i == n_strata:
                has_unseen_strata = True

    return output, has_unseen_strata
