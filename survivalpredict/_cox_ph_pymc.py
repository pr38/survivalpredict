from typing import Literal, Optional

import numpy as np


def get_cox_pymc_model_no_strata(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    time_end_return_inverse: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    n_unique_times: int,
    time_start_return_inverse: Optional[
        np.ndarray[tuple[int], np.dtype[np.int64]]
    ] = None,
    ties: Literal["efron", "breslow"] = "breslow",
    column_names: Optional[list[str]] = None,
    l_div_m: Optional[np.ndarray[tuple[int], np.dtype[np.float64]]] = None,
    coefs_sigma: float = 10.0,
    alpha: float = 0.0,
    coefs_inital: Optional[np.ndarray[tuple[int], np.dtype[np.float64]]] = None,
) -> "pymc.Model":

    # lazy imports
    import pymc as pm
    import pytensor.tensor as pt

    if l_div_m is None and ties == "efron":
        raise ValueError("efron ties need a precalcuated l_div_m array")

    used_left_censorship = time_start_return_inverse is not None

    coords = {}
    if column_names is not None:
        coords["labes"] = column_names
    else:
        coords["labes"] = np.arange(X.shape[1])


    with pm.Model(coords=coords) as model:
        data = pm.Data("data", X)
        events_pt = pm.Data("events", events)

        if ties == "efron":
            l_div_m_pt = pm.Data("l_div_m", l_div_m)

        coefs = pm.Normal("coefs", 0, coefs_sigma, dims="labes",initval=coefs_inital)

        ###getting products
        o = pt.dot(data, coefs)
        o_exp = pt.exp(o)

        ###getting risk set
        if used_left_censorship:
            risk_removed_at_time = pt.bincount(
                time_end_return_inverse, weights=o_exp, minlength=n_unique_times
            )
            risk_added_at_time = pt.bincount(
                time_start_return_inverse, weights=o_exp, minlength=n_unique_times
            )
            risk_at_time = pt.cumsum(risk_added_at_time - risk_removed_at_time)
            risk_set = risk_at_time[time_end_return_inverse - 1]
        else:
            risk_set_at_time = pt.flip(
                pt.cumsum(
                    pt.flip(
                        pt.bincount(
                            time_end_return_inverse,
                            weights=o_exp,
                            minlength=n_unique_times,
                        )
                    )
                )
            )
            risk_set = risk_set_at_time[time_end_return_inverse]

        ###getting likelihood
        if ties == "efron":
            total_event_risk = pt.bincount(
                time_end_return_inverse,
                weights=o_exp * events_pt,
                minlength=n_unique_times,
            )[time_end_return_inverse]
            partial_log_likelihood = pt.sum(
                events_pt
                * (
                    o
                    - pt.nan_to_num(pt.log(risk_set - (l_div_m_pt * total_event_risk)))
                )
            )

        else:  # ties == 'breslow'
            partial_log_likelihood = pt.sum(
                events_pt * (o - pt.nan_to_num(pt.log(risk_set)))
            )

        if alpha > 0.0:  # add elastic net loss
            l2 = 0.5 * alpha * pt.square(coefs).sum()
            partial_log_likelihood = partial_log_likelihood - l2

        pm.Deterministic("n_log_likelihood", -partial_log_likelihood)

        pm.Potential("loss", partial_log_likelihood)

    return model


def _get_breslow_likelihood_per_strata(
    strata_mask, o, o_exp, time_end_return_inverse, n_unique_times, events_pt
):
    # lazy imports
    import pytensor.tensor as pt

    o = o[strata_mask]
    o_exp = o_exp[strata_mask]
    time_end_return_inverse = time_end_return_inverse[strata_mask]
    events_pt = events_pt[strata_mask]

    risk_set_at_time = pt.flip(
        pt.cumsum(
            pt.flip(
                pt.bincount(
                    time_end_return_inverse, weights=o_exp, minlength=n_unique_times
                )
            )
        )
    )
    risk_set = risk_set_at_time[time_end_return_inverse]

    partial_log_likelihood = pt.sum(events_pt * (o - pt.nan_to_num(pt.log(risk_set))))

    return partial_log_likelihood


def _get_breslow_likelihood_with_left_cenorship_per_strata(
    strata_mask,
    o,
    o_exp,
    time_end_return_inverse,
    time_start_return_inverse,
    n_unique_times,
    events_pt,
):
    # lazy imports
    import pytensor.tensor as pt

    o = o[strata_mask]
    o_exp = o_exp[strata_mask]
    time_end_return_inverse = time_end_return_inverse[strata_mask]
    time_start_return_inverse = time_start_return_inverse[strata_mask]
    events_pt = events_pt[strata_mask]

    risk_removed_at_time = pt.bincount(
        time_end_return_inverse, weights=o_exp, minlength=n_unique_times
    )
    risk_added_at_time = pt.bincount(
        time_start_return_inverse, weights=o_exp, minlength=n_unique_times
    )
    risk_at_time = pt.cumsum(risk_added_at_time - risk_removed_at_time)
    risk_set = risk_at_time[time_end_return_inverse - 1]

    partial_log_likelihood = pt.sum(events_pt * (o - pt.nan_to_num(pt.log(risk_set))))

    return partial_log_likelihood


def _get_efron_likelihood_per_strata(
    strata_mask, o, o_exp, time_end_return_inverse, l_div_m, n_unique_times, events_pt
):
    # lazy imports
    import pytensor.tensor as pt

    o = o[strata_mask]
    o_exp = o_exp[strata_mask]
    l_div_m = l_div_m[strata_mask]
    time_end_return_inverse = time_end_return_inverse[strata_mask]
    events_pt = events_pt[strata_mask]

    risk_set_at_time = pt.flip(
        pt.cumsum(
            pt.flip(
                pt.bincount(
                    time_end_return_inverse, weights=o_exp, minlength=n_unique_times
                )
            )
        )
    )
    risk_set = risk_set_at_time[time_end_return_inverse]

    total_event_risk = pt.bincount(
        time_end_return_inverse,
        weights=o_exp * events_pt,
        minlength=n_unique_times,
    )[time_end_return_inverse]

    partial_log_likelihood = pt.sum(
        events_pt * (o - pt.nan_to_num(pt.log(risk_set - (l_div_m * total_event_risk))))
    )

    return partial_log_likelihood


def _get_efron_likelihood_with_left_cenorship_per_strata(
    strata_mask,
    o,
    o_exp,
    time_end_return_inverse,
    time_start_return_inverse,
    l_div_m,
    n_unique_times,
    events_pt,
):
    # lazy imports
    import pytensor.tensor as pt

    o = o[strata_mask]
    o_exp = o_exp[strata_mask]
    l_div_m = l_div_m[strata_mask]
    time_end_return_inverse = time_end_return_inverse[strata_mask]
    time_start_return_inverse = time_start_return_inverse[strata_mask]
    events_pt = events_pt[strata_mask]

    risk_removed_at_time = pt.bincount(
        time_end_return_inverse, weights=o_exp, minlength=n_unique_times
    )
    risk_added_at_time = pt.bincount(
        time_start_return_inverse, weights=o_exp, minlength=n_unique_times
    )
    risk_at_time = pt.cumsum(risk_added_at_time - risk_removed_at_time)
    risk_set = risk_at_time[time_end_return_inverse - 1]

    total_event_risk = pt.bincount(
        time_end_return_inverse,
        weights=o_exp * events_pt,
        minlength=n_unique_times,
    )[time_end_return_inverse]

    partial_log_likelihood = pt.sum(
        events_pt * (o - pt.nan_to_num(pt.log(risk_set - (l_div_m * total_event_risk))))
    )

    return partial_log_likelihood


def get_cox_pymc_model_with_strata(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    time_end_return_inverse: np.ndarray[tuple[int], np.dtype[np.int64]],
    events: np.ndarray[tuple[int], np.dtype[np.bool_]],
    strata: np.ndarray[tuple[int], np.dtype[np.int64]],
    n_strata: int,
    n_unique_times: int,
    time_start_return_inverse: Optional[
        np.ndarray[tuple[int], np.dtype[np.int64]]
    ] = None,
    ties: Literal["efron", "breslow"] = "breslow",
    column_names: Optional[list[str]] = None,
    l_div_m: Optional[np.ndarray[tuple[int], np.dtype[np.float64]]] = None,
    coefs_sigma: float = 10.0,
    alpha: float = 0.0,
    coefs_inital: Optional[np.ndarray[tuple[int], np.dtype[np.float64]]] = None,
) -> "pymc.Model":
    # lazy imports
    import pymc as pm
    import pytensor.tensor as pt
    import pytensor

    if l_div_m is None and ties == "efron":
        raise ValueError("efron ties need a precalcuated l_div_m array")

    in_strata_mask = strata[:, None] == np.arange(n_strata)

    used_left_censorship = time_start_return_inverse is not None

    coords = {}
    if column_names is not None:
        coords["labes"] = column_names
    else:
        coords["labes"] = np.arange(X.shape[1])

    with pm.Model(coords=coords) as model:

        data = pm.Data("data", X)
        events_pt = pm.Data("events", events)
        time_end_return_inverse_pt = pm.Data(
            "time_end_return_inverse", time_end_return_inverse
        )
        n_unique_times_pt = pm.Data("n_unique_times", np.array(n_unique_times))
        in_strata_mask_T_pt = pm.Data("in_strata_mask_T", in_strata_mask.T)

        if ties == "efron":
            l_div_m_pt = pm.Data("l_div_m", l_div_m)
        if used_left_censorship:
            time_start_return_inverse_pt = pm.Data(
                "time_start_return_inverse", time_start_return_inverse
            )

        coefs = pm.Normal("coefs", 0, coefs_sigma, dims="labes",initval=coefs_inital)

        o = pt.dot(data, coefs)
        o_exp = pt.exp(o)

        if used_left_censorship is False:
            if ties == "efron":
                likelihood_per_strata, _ = pytensor.scan(
                    _get_efron_likelihood_per_strata,
                    in_strata_mask_T_pt,
                    non_sequences=[
                        o,
                        o_exp,
                        time_end_return_inverse_pt,
                        l_div_m_pt,
                        n_unique_times_pt,
                        events_pt,
                    ],
                )
            else:
                likelihood_per_strata, _ = pytensor.scan(
                    _get_breslow_likelihood_per_strata,
                    in_strata_mask_T_pt,
                    non_sequences=[
                        o,
                        o_exp,
                        time_end_return_inverse_pt,
                        n_unique_times_pt,
                        events_pt,
                    ],
                )
        else:
            if ties == "efron":
                likelihood_per_strata, _ = pytensor.scan(
                    _get_efron_likelihood_with_left_cenorship_per_strata,
                    in_strata_mask_T_pt,
                    non_sequences=[
                        o,
                        o_exp,
                        time_end_return_inverse_pt,
                        time_start_return_inverse_pt,
                        l_div_m_pt,
                        n_unique_times_pt,
                        events_pt,
                    ],
                )
            else:
                likelihood_per_strata, _ = pytensor.scan(
                    _get_breslow_likelihood_with_left_cenorship_per_strata,
                    in_strata_mask_T_pt,
                    non_sequences=[
                        o,
                        o_exp,
                        time_end_return_inverse_pt,
                        time_start_return_inverse_pt,
                        n_unique_times_pt,
                        events_pt,
                    ],
                )

        partial_log_likelihood = pt.sum(likelihood_per_strata)

        if alpha > 0.0:  # add elastic net loss
            l2 = 0.5 * alpha * pt.square(coefs).sum()
            partial_log_likelihood = partial_log_likelihood - l2

        pm.Deterministic("n_log_likelihood", -partial_log_likelihood)

        pm.Potential("loss", partial_log_likelihood)

    return model
