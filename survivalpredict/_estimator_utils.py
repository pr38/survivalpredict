from collections import Counter
from typing import Any, Iterable, Optional

import numpy as np

from ._data_validation import _as_bool_np_array, _as_int_np_array


def _get_estimator_names(estimators: Iterable[Any]) -> list[str]:
    "stripped down vestion of sklearn/pipeline's _name_estimators"
    names = [type(e).__name__.lower() for e in estimators]

    namecount = Counter(names)

    for k, v in list(namecount.items()):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(names))):
        name = names[i]

        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1
    return names


def _unpack_sklearn_pipeline_target(
    y: np.ndarray,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.integer]],
    np.ndarray[tuple[int], np.dtype[np.bool_]],
    Optional[np.ndarray[tuple[int], np.dtype[np.integer]]],
]:
    "Sister function for build_sklearn_pipeline_target"
    times = _as_int_np_array(y["times"])
    events = _as_bool_np_array(y["events"])

    if "strata" in y.dtype.names:
        strata = _as_int_np_array(y["strata"], "strata")
        return times, events, strata
    else:
        return times, events, None
