import numba as nb
import numpy as np
from sklearn.tree._tree import Tree, _build_pruned_tree_ccp


@nb.njit(
    nb.types.Array(nb.types.float64, 1, "C", False, aligned=True)(
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
    ),
    cache=True,
)
def get_km_curve_from_counts(
    death_per_step: np.ndarray[tuple[int], np.dtype[np.float64]],
    exit_per_step: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    right_censor_count_per_step = exit_per_step - death_per_step

    at_risk_per_step = np.flip(
        np.cumsum(np.flip(right_censor_count_per_step + death_per_step))
    )

    hazard_at_step = np.divide(
        death_per_step,
        at_risk_per_step,
    )
    hazard_at_step = np.where(death_per_step != 0, hazard_at_step, 0)
    return (1 - hazard_at_step).cumprod()


@nb.njit(
    nb.types.float64(
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.int64,
    ),
    cache=True,
)
def get_integrated_brier_score_administrative_of_km_curve_from_counts(
    death_per_step: np.ndarray[tuple[int], np.dtype[np.float64]],
    exit_per_step: np.ndarray[tuple[int], np.dtype[np.float64]],
    km_curve: np.ndarray[tuple[int], np.dtype[np.float64]],
    max_time: int,
) -> float:

    max_index = max_time + 1

    counts_dead_at_step = np.cumsum(death_per_step[1:max_index])
    death_scores = counts_dead_at_step * np.square(km_curve[1:max_index])

    counts_alive_at_step = exit_per_step.sum() - np.cumsum(exit_per_step[1:max_index])
    alive_scores = counts_alive_at_step * np.square(1 - km_curve[1:max_index])

    scores = (death_scores + alive_scores) / (
        counts_dead_at_step + counts_alive_at_step
    )

    # return np.trapezoid(np.nan_to_num(scores))
    return np.sum(np.nan_to_num(scores))


@nb.njit(
    nb.types.float64(
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.int64,
    ),
    cache=True,
)
def wasserstein_distance_impurity(
    death_per_step_left: np.ndarray[tuple[int], np.dtype[np.float64]],
    exit_per_step_left: np.ndarray[tuple[int], np.dtype[np.float64]],
    death_per_step_right: np.ndarray[tuple[int], np.dtype[np.float64]],
    exit_per_step_right: np.ndarray[tuple[int], np.dtype[np.float64]],
    max_time: int,
) -> float:
    left_km = get_km_curve_from_counts(death_per_step_left, exit_per_step_left)

    right_km = get_km_curve_from_counts(death_per_step_right, exit_per_step_right)

    return np.sum(
        np.abs((1 - left_km[: max_time + 1]) - (1 - right_km[: max_time + 1]))
    )


@nb.njit(
    nb.types.float64(
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
    ),
    cache=True,
)
def log_rank_approximate_impurity(
    in_risk_set, death_per_step, death_per_step_left, events_per_step_left
):
    in_risk_set_left = np.flip(np.cumsum(np.flip(events_per_step_left)))
    expected = in_risk_set_left / in_risk_set * death_per_step
    observed_minus_expected = death_per_step_left - expected
    return np.nan_to_num(observed_minus_expected**2 / expected).sum()


@nb.njit(
    nb.types.float64(
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.int64,
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.int64,
        nb.types.int64,
    ),
    cache=True,
)
def integrated_brier_score_administrative_impurity(
    death_per_step_left: np.ndarray[tuple[int], np.dtype[np.float64]],
    exit_per_step_left: np.ndarray[tuple[int], np.dtype[np.float64]],
    left_weights: float,
    death_per_step_right: np.ndarray[tuple[int], np.dtype[np.float64]],
    exit_per_step_right: np.ndarray[tuple[int], np.dtype[np.float64]],
    right_weights: float,
    max_time: int,
) -> float:
    left_km_curve = get_km_curve_from_counts(death_per_step_left, exit_per_step_left)
    left_score = get_integrated_brier_score_administrative_of_km_curve_from_counts(
        death_per_step_left, exit_per_step_left, left_km_curve, max_time
    )

    right_km_curve = get_km_curve_from_counts(death_per_step_right, exit_per_step_right)
    right_score = get_integrated_brier_score_administrative_of_km_curve_from_counts(
        death_per_step_right, exit_per_step_right, right_km_curve, max_time
    )

    return -(right_score / right_weights + left_score / left_weights / 2)


@nb.njit(
    nb.types.Tuple((nb.types.float64, nb.types.float64))(
        nb.types.Array(nb.types.float64, 1, "A", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.bool_, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.int64,
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.int64,
        nb.types.int64,
        nb.types.int64,
        nb.types.float64,
    ),
    cache=True,
)
def get_best_threshold_on_col(
    col: np.ndarray[tuple[int], np.dtype[np.float64]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.int64]],
    weights: np.ndarray[tuple[int], np.dtype[np.float64]],
    max_time: int,
    death_per_step: np.ndarray[tuple[int], np.dtype[np.int64]],
    events_per_step: np.ndarray[tuple[int], np.dtype[np.int64]],
    n_rows: int,
    min_samples_leaf: int,
    crit_code: int,
    weights_total: float,
):
    best_impurity = -np.inf
    best_threshold = -1.0

    arg_sort = np.argsort(col)
    times_sort = times[arg_sort]
    col_sort = col[arg_sort]
    events_sort = events[arg_sort]
    weights_sort = weights[arg_sort]
    last_value = col_sort[-1]

    death_per_step_right = death_per_step.copy()
    exit_per_step_right = events_per_step.copy()
    weights_right = weights_total

    death_per_step_left = np.zeros(max_time + 1)
    exit_per_step_left = np.zeros(max_time + 1)

    for row_index in range(n_rows - 1):

        value_i = col_sort[row_index]
        times_i = times_sort[row_index]
        event_i = events_sort[row_index]
        weight_i = weights_sort[row_index]

        if event_i == 1:
            death_per_step_right[times_i] -= weight_i
            death_per_step_left[times_i] += weight_i

        exit_per_step_right[times_i] -= weight_i
        exit_per_step_left[times_i] += weight_i

        weights_right -= weight_i

        if crit_code == 2:
            at_risk_per_step = np.flip(np.cumsum(np.flip(events_per_step))).astype(
                np.float64
            )

        if (value_i != col_sort[row_index + 1]) and (value_i != last_value):
            # left_size = row_index + 1
            # right_size = n_rows - left_size
            weights_left = weights_total - weights_right

            if (
                # (left_size >= min_samples_leaf)
                # and (right_size >= min_samples_leaf)
                (weights_left >= min_samples_leaf)
                and (weights_right >= min_samples_leaf)
            ):

                if crit_code == 0:
                    impurity = wasserstein_distance_impurity(
                        death_per_step_left,
                        exit_per_step_left,
                        death_per_step_right,
                        exit_per_step_right,
                        max_time,
                    )

                elif crit_code == 1:
                    impurity = integrated_brier_score_administrative_impurity(
                        death_per_step_left,
                        exit_per_step_left,
                        weights_left,
                        death_per_step_right,
                        exit_per_step_right,
                        weights_right,
                        max_time,
                    )

                else:
                    impurity = log_rank_approximate_impurity(
                        at_risk_per_step,
                        death_per_step,
                        death_per_step_left,
                        exit_per_step_left,
                    )

                if impurity > best_impurity:
                    best_impurity = impurity
                    best_threshold = (value_i + col_sort[row_index + 1]) / 2

    return best_threshold, best_impurity


@nb.njit(
    nb.types.Tuple((nb.types.int64, nb.types.float64, nb.types.float64))(
        nb.types.Array(nb.types.float64, 2, "C", False, aligned=True),
        nb.types.Array(nb.types.int64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.bool_, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.types.float64, 1, "C", False, aligned=True),
        nb.types.int64,
        nb.types.int64,
        nb.types.int64,
        nb.types.int64,
        nb.types.int64,
    ),
    cache=True,
    # parallel=True,
)
def get_best_threshold_on_data(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    death_per_step: np.ndarray[tuple[int], np.dtype[np.float64]],
    events_per_step: np.ndarray[tuple[int], np.dtype[np.float64]],
    weights: np.ndarray[tuple[int], np.dtype[np.float64]],
    min_samples_leaf: int,
    max_features: int,
    random_state: int,
    crit_code: int,
    max_time: int,
) -> tuple[int, float, float]:
    if random_state >= 0:
        np.random.seed(random_state)

    n_rows, n_cols = X.shape

    if max_features < 0:  # if max_features is not being used, examine all cols
        cols_to_examine = np.arange(n_cols)
    else:
        cols_to_examine = np.random.choice(n_cols, max_features, replace=False)

    best_col_proxy_impurites = np.full(
        shape=n_cols, fill_value=-np.inf, dtype=np.float64
    )
    best_col_thresholds = np.zeros(n_cols)
    weights_total = weights.sum()

    for col_index in nb.prange(n_cols):
        col = X[:, col_index]
        if col_index in cols_to_examine:
            best_threshold, best_proxy_impurity = get_best_threshold_on_col(
                col,
                times,
                events,
                weights,
                max_time,
                death_per_step,
                events_per_step,
                n_rows,
                min_samples_leaf,
                crit_code,
                weights_total,
            )

            best_col_proxy_impurites[col_index] = best_proxy_impurity
            best_col_thresholds[col_index] = best_threshold

    best_cols_bool = best_col_proxy_impurites == max(best_col_proxy_impurites)
    best_col = int(np.random.choice(np.flatnonzero(best_cols_bool)))
    # if 2 or more col share the best approximation, pick one at random.
    # This is done to immitate sklearn behaviour(see the splitter classes in the tree modual).

    best_threshold = best_col_thresholds[best_col]
    best_impurity = best_col_proxy_impurites[best_col]

    return best_col, best_threshold, best_impurity


@nb.njit(
    nb.types.Tuple(
        (
            nb.types.Array(nb.float64, 2, "C", False, aligned=True),
            nb.types.Array(nb.int64, 1, "C", False, aligned=True),
            nb.types.Array(nb.bool_, 1, "C", False, aligned=True),
            nb.types.Array(nb.float64, 1, "C", False, aligned=True),
            nb.types.Array(nb.float64, 2, "C", False, aligned=True),
            nb.types.Array(nb.int64, 1, "C", False, aligned=True),
            nb.types.Array(nb.bool_, 1, "C", False, aligned=True),
            nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        )
    )(
        nb.types.Array(nb.float64, 2, "C", False, aligned=True),
        nb.types.Array(nb.int64, 1, "C", False, aligned=True),
        nb.types.Array(nb.bool_, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.float64,
        nb.int64,
    ),
    cache=True,
    # parallel=True,
)
def split_data(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    weights: np.ndarray[tuple[int], np.dtype[np.float64]],
    threshold: float,
    feature: int,
):

    left_mask_split = X[:, feature] <= threshold
    right_mask_split = ~left_mask_split

    right_X = X[right_mask_split]
    right_times = times[right_mask_split]
    right_events = events[right_mask_split]
    right_weights = weights[right_mask_split]

    left_X = X[left_mask_split]
    left_times = times[left_mask_split]
    left_events = events[left_mask_split]
    left_weights = weights[left_mask_split]

    return (
        left_X,
        left_times,
        left_events,
        left_weights,
        right_X,
        right_times,
        right_events,
        right_weights,
    )


@nb.njit(
    nb.types.Tuple(
        (
            nb.types.boolean,
            nb.types.boolean,
            nb.types.int64,
            nb.types.float64,
            nb.types.int64,
            nb.types.float64,
            nb.types.optional(nb.types.Array(nb.float64, 2, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.int64, 1, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.bool_, 1, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.types.optional(nb.float64),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.float64, 2, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.int64, 1, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.bool_, 1, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.types.optional(nb.float64),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.types.optional(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
        )
    )(
        nb.types.Array(nb.float64, 2, "C", False, aligned=True),
        nb.types.Array(nb.int64, 1, "C", False, aligned=True),
        nb.types.Array(nb.bool_, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.types.boolean,
        nb.types.int64,
        nb.types.float64,
        nb.types.int64,
        nb.types.int64,
        nb.types.int64,
        nb.types.int64,
        nb.types.float64,
        nb.types.float64,
        nb.types.int64,
        nb.types.int64,
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.types.int64,
    ),
    cache=True,
)
def process_node(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    weights: np.ndarray[tuple[int], np.dtype[np.float64]],
    is_left: bool,
    depth: int,
    score: float,
    max_depth_limit: int,
    min_samples_leaf: int,
    min_samples_split: int,
    crit_code: int,
    min_impurity_decrease: float,
    min_weight_leaf: float,
    max_features: int,
    random_state: int,
    death_per_step: np.ndarray[tuple[int], np.dtype[np.int64]],
    events_per_step: np.ndarray[tuple[int], np.dtype[np.int64]],
    max_time: int,
):
    minlength = max_time + 1

    n_node_sample = X.shape[0]
    weighted_n_node_sample = weights.sum()

    exceeds_depth = False if max_depth_limit <= 0 else depth >= max_depth_limit

    is_leaf = (
        exceeds_depth
        or n_node_sample < min_samples_split
        or n_node_sample < 2 * min_samples_leaf
        or weighted_n_node_sample < 2 * min_weight_leaf
    )

    is_leaf = is_leaf or score <= np.finfo(np.float64).eps

    if not is_leaf:
        feature, threshold, impurity = get_best_threshold_on_data(
            X,
            times,
            events,
            death_per_step,
            events_per_step,
            weights,
            min_samples_leaf,
            max_features,
            random_state,
            crit_code,
            max_time,
        )

        if impurity == -np.inf:
            is_leaf = True

    if not is_leaf:
        (
            left_X,
            left_times,
            left_events,
            left_weights,
            right_X,
            right_times,
            right_events,
            right_weights,
        ) = split_data(X, times, events, weights, threshold, feature)

        left_death_per_step = np.bincount(
            left_times, left_events * left_weights, minlength=minlength
        ).astype(np.float64)[: max_time + 1]
        left_events_per_step = np.bincount(
            left_times, left_weights, minlength=minlength
        ).astype(np.float64)[: max_time + 1]

        left_km_curve = get_km_curve_from_counts(
            left_death_per_step, left_events_per_step
        )
        left_score = get_integrated_brier_score_administrative_of_km_curve_from_counts(
            left_death_per_step, left_events_per_step, left_km_curve, max_time
        )

        right_death_per_step = np.bincount(
            right_times, right_events * right_weights, minlength=minlength
        ).astype(np.float64)[: max_time + 1]
        right_events_per_step = np.bincount(
            right_times, right_weights, minlength=minlength
        ).astype(np.float64)[: max_time + 1]

        right_km_curve = get_km_curve_from_counts(
            right_death_per_step, right_events_per_step
        )
        right_score = get_integrated_brier_score_administrative_of_km_curve_from_counts(
            right_death_per_step, right_events_per_step, right_km_curve, max_time
        )

        new_score = (right_score / len(right_times) + left_score / len(left_times)) / 2

        improvement = score - new_score

        if (improvement + np.finfo(np.float64).eps) < min_impurity_decrease:
            is_leaf = True

    if is_leaf:
        threshold = -2
        feature = -2

        left_X = None
        left_times = None
        left_events = None
        left_weights = None
        left_score = None
        left_death_per_step = None
        left_events_per_step = None
        left_km_curve = None
        right_X = None
        right_times = None
        right_events = None
        right_weights = None
        right_score = None
        right_death_per_step = None
        right_events_per_step = None
        right_km_curve = None

    return (
        is_leaf,
        is_left,
        feature,
        threshold,
        n_node_sample,
        weighted_n_node_sample,
        left_X,
        left_times,
        left_events,
        left_weights,
        left_score,
        left_death_per_step,
        left_events_per_step,
        left_km_curve,
        right_X,
        right_times,
        right_events,
        right_weights,
        right_score,
        right_death_per_step,
        right_events_per_step,
        right_km_curve,
    )


@nb.njit(
    nb.int64(
        nb.types.List(
            nb.types.Tuple(
                (
                    nb.int64,
                    nb.boolean,
                    nb.types.Array(nb.float64, 2, "C", False, aligned=True),
                    nb.types.Array(nb.int64, 1, "C", False, aligned=True),
                    nb.types.Array(nb.bool_, 1, "C", False, aligned=True),
                    nb.types.Array(nb.float64, 1, "C", False, aligned=True),
                    nb.types.Array(nb.float64, 1, "C", False, aligned=True),
                    nb.types.Array(nb.float64, 1, "C", False, aligned=True),
                    nb.types.Array(nb.float64, 1, "C", False, aligned=True),
                    nb.int64,
                    nb.float64,
                )
            )
        )
    ),
    inline="always",
    cache=True,
)
def get_index_with_highest_score(to_build_stack):
    best_score = np.inf
    best_index = -1

    for index, to_build_node in enumerate(to_build_stack):
        score = to_build_node[-1]  # score is value
        if score < best_score:
            best_score = score
            best_index = index

    return best_index


@nb.njit(
    nb.types.Tuple(
        (
            nb.types.List(nb.int64),
            nb.types.List(nb.int64),
            nb.types.List(nb.int64),
            nb.types.List(nb.float64),
            nb.types.List(nb.float64),
            nb.types.List(nb.int64),
            nb.types.List(nb.float64),
            nb.types.List(nb.types.Array(nb.float64, 1, "C", False, aligned=True)),
            nb.int64,
        )
    )(
        nb.types.Array(nb.float64, 2, "C", False, aligned=True),
        nb.types.Array(nb.int64, 1, "C", False, aligned=True),
        nb.types.Array(nb.bool_, 1, "C", False, aligned=True),
        nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.float64,
        nb.float64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
    ),
    cache=True,
)
def build_tree(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    weights: np.ndarray[tuple[int], np.dtype[np.float64]],
    min_samples_leaf: int,
    max_depth_limit: int,
    min_samples_split: int,
    crit_code: int,
    min_impurity_decrease: float,
    min_weight_leaf: float,
    max_features: int,
    max_leaf_nodes: int,
    random_state: int,
    max_time: int,
):
    left_childs = []
    right_childs = []
    features = []
    thresholds = []
    scores = []
    n_node_samples = []
    weighted_n_node_samples = []
    values = []

    root_death_per_step = np.bincount(times, events * weights, minlength=max_time + 1)[
        : max_time + 1
    ]
    root_exit_per_step = np.bincount(times, weights, minlength=max_time + 1)[
        : max_time + 1
    ]

    km_curve = get_km_curve_from_counts(root_death_per_step, root_exit_per_step)
    score = get_integrated_brier_score_administrative_of_km_curve_from_counts(
        root_death_per_step, root_exit_per_step, km_curve, max_time
    )

    to_build_stack = nb.typed.List()

    to_build_stack.append(
        (
            -1,
            True,
            X,
            times,
            events,
            weights,
            root_death_per_step,
            root_exit_per_step,
            km_curve,
            0,
            score,
        )
    )  # parentid,is_left,X,times,events,,weights,depth,score

    current_index = 0
    depth = 0
    max_depth_build = 0
    weighted_n_samples = weights.sum()
    leaf_count = 0

    while len(to_build_stack) > 0:

        # if max_leaf_nodes is being used, get index of best nodes first
        if max_leaf_nodes < 0:
            index_to_pop = -1
        else:
            index_to_pop = get_index_with_highest_score(to_build_stack)

        # get from stack
        (
            parent_index,
            is_left,
            X,
            times,
            events,
            weights,
            death_per_step,
            events_per_step,
            km_curve,
            depth,
            score,
        ) = to_build_stack.pop(index_to_pop)

        # process node
        (
            is_leaf,
            is_left,
            feature,
            threshold,
            n_node_sample,
            weighted_n_node_sample,
            left_X,
            left_times,
            left_events,
            left_weights,
            left_score,
            left_death_per_step,
            left_events_per_step,
            left_km_curve,
            right_X,
            right_times,
            right_events,
            right_weights,
            right_score,
            right_death_per_step,
            right_events_per_step,
            right_km_curve,
        ) = process_node(
            X,
            times,
            events,
            weights,
            is_left,
            depth,
            score,
            max_depth_limit,
            min_samples_leaf,
            min_samples_split,
            crit_code,
            min_impurity_decrease,
            min_weight_leaf,
            max_features,
            random_state,
            death_per_step,
            events_per_step,
            max_time,
        )

        # add to tree
        left_childs.append(-1)
        right_childs.append(-1)
        features.append(feature)
        values.append(km_curve)
        thresholds.append(threshold)
        scores.append(score)
        n_node_samples.append(n_node_sample)
        weighted_n_node_samples.append(weighted_n_node_sample)

        if parent_index != -1:  # skip the root/first node
            if is_left:
                left_childs[parent_index] = current_index
            else:
                right_childs[parent_index] = current_index

        if not is_leaf:
            to_build_stack.append(
                (
                    current_index,
                    False,
                    right_X,
                    right_times,
                    right_events,
                    right_weights,
                    right_death_per_step,
                    right_events_per_step,
                    right_km_curve,
                    depth + 1,
                    right_score,
                )
            )
            to_build_stack.append(
                (
                    current_index,
                    True,
                    left_X,
                    left_times,
                    left_events,
                    left_weights,
                    left_death_per_step,
                    left_events_per_step,
                    left_km_curve,
                    depth + 1,
                    left_score,
                )
            )
        else:
            leaf_count += 1

        # book keeping, for future nodes
        current_index = current_index + 1

        if depth > max_depth_build and not is_leaf:
            max_depth_build = depth

        # if max_leaf_nodes is being used, stop tree building if max_leaf_nodes limit is hit
        if max_leaf_nodes > 0:
            if leaf_count >= max_leaf_nodes:
                to_build_stack.clear()

    return (
        left_childs,
        right_childs,
        features,
        thresholds,
        scores,
        n_node_samples,
        weighted_n_node_samples,
        values,
        max_depth_build,
    )


def prune_sk_Tree_class(tree, n_features, n_classes, ccp_alpha):
    pruned_tree = Tree(n_features, np.array([n_classes]), 1)
    _build_pruned_tree_ccp(pruned_tree, tree, ccp_alpha)
    return pruned_tree


def get_survival_tree(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    weights: np.ndarray[tuple[int], np.dtype[np.float64]],
    min_samples_leaf: int,
    max_depth_limit: int,
    min_samples_split: int,
    crit_code: int,
    min_impurity_decrease: float,
    min_weight_leaf: float,
    max_features: int,
    max_leaf_nodes: int,
    random_state: int,
    max_time: int,
    ccp_alpha,
) -> Tree:
    (
        left_childs,
        right_childs,
        features,
        thresholds,
        scores,
        n_node_samples,
        weighted_n_node_samples,
        values,
        max_depth_build,
    ) = build_tree(
        X,
        times,
        events,
        weights,
        min_samples_leaf,
        max_depth_limit,
        min_samples_split,
        crit_code,
        min_impurity_decrease,
        min_weight_leaf,
        max_features,
        max_leaf_nodes,
        random_state,
        max_time,
    )

    dt = {
        "names": [
            "left_child",
            "right_child",
            "feature",
            "threshold",
            "impurity",
            "n_node_samples",
            "weighted_n_node_samples",
            "missing_go_to_left",
        ],
        "formats": ["<i8", "<i8", "<i8", "<f8", "<f8", "<i8", "<f8", "u1"],
    }
    nodes = np.zeros(len(left_childs), dtype=dt)
    nodes["left_child"] = left_childs
    nodes["right_child"] = right_childs
    nodes["feature"] = features
    nodes["threshold"] = thresholds
    nodes["impurity"] = scores
    nodes["n_node_samples"] = n_node_samples
    nodes["weighted_n_node_samples"] = weighted_n_node_samples
    nodes["missing_go_to_left"] = np.zeros(len(left_childs), dtype=np.int64)

    state = {
        "max_depth": max_depth_build,
        "node_count": len(features),
        "nodes": nodes,
        "values": np.array(values)[:, np.newaxis, :],
    }

    tree = Tree(X.shape[1], np.array([len(values[0])]), 1)
    tree.__setstate__(state)

    n_features = X.shape[0]
    n_classes = len(values[0])

    tree = prune_sk_Tree_class(
        tree=tree, n_features=n_features, n_classes=n_classes, ccp_alpha=ccp_alpha
    )

    return tree
