import numpy as np
from sklearn.utils import check_random_state


from numbers import Integral
from warnings import warn


def get_n_samples_bootstrap(n_samples, max_samples, sample_weight):
    """
    Taken from sklean.ensemble._bootstrap.
    Copied over here to avoid future upstream breaking changes.
    """
    if max_samples is None:
        return n_samples
    elif isinstance(max_samples, Integral):
        return max_samples

    if sample_weight is None:
        weighted_n_samples = n_samples
        weighted_n_samples_msg = f"the number of samples is {weighted_n_samples} "
    else:
        weighted_n_samples = sample_weight.sum()
        weighted_n_samples_msg = (
            f"the total sum of sample weights is {weighted_n_samples} "
        )

    n_samples_bootstrap = max(int(max_samples * weighted_n_samples), 1)

    if n_samples_bootstrap < max(10, n_samples ** (1 / 3)):
        warn(
            f"Using the fractional value {max_samples=} when {weighted_n_samples_msg}"
            f"results in a low number ({n_samples_bootstrap}) of bootstrap samples. "
            "We recommend passing `max_samples` as an integer instead."
        )
    return n_samples_bootstrap


def _generate_sample_indices(
    random_state, n_samples, n_samples_bootstrap, sample_weight
):
    """
    Private function used to _parallel_build_trees function.
    Taken from sklearn.
    """

    random_instance = check_random_state(random_state)
    if sample_weight is None:
        sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
    else:
        normalized_sample_weight = sample_weight / np.sum(sample_weight)
        sample_indices = random_instance.choice(
            n_samples,
            n_samples_bootstrap,
            replace=True,
            p=normalized_sample_weight,
        )
    sample_indices = sample_indices.astype(np.int32)
    return sample_indices


def build_tree(
    tree,
    bootstrap,
    X,
    times,
    events,
    sample_weight,
    n_samples_bootstrap=None,
    times_start=None,
):
    """
    Private function used to fit a single tree in parallel.
    Converted from sklearn.
    """

    if bootstrap:
        n_samples = X.shape[0]
        indices = _generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap, sample_weight
        )
        # Simulate row-wise sampling by passing counts as sample_weight in trees.
        sample_weight_tree = np.bincount(indices, minlength=n_samples).astype(
            np.float64
        )

        tree.fit(
            X,
            times,
            events,
            sample_weight=sample_weight_tree,
            check_input=False,
            times_start=times_start
        )
    else:
        tree._fit(
            X,
            times,
            events,
            sample_weight=sample_weight,
            check_input=False,
            times_start=times_start
        )

    return tree
