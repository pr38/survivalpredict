from typing import Iterable, Any
from collections import Counter

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
