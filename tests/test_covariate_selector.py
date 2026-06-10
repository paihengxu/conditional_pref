from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
WIMHF_REPO = PROJECT_DIR / "repos" / "wimhf"
if str(WIMHF_REPO) not in sys.path:
    sys.path.insert(0, str(WIMHF_REPO))

from wimhf.feature_selection import (
    _compute_equal_stratum_sample_weights,
    _demean_by_covariate_strata,
    select_neurons_demeaned_reweighted_lasso,
)


def test_equal_stratum_weights_have_equal_mass_and_mean_one() -> None:
    covariates = np.array([[0], [0], [0], [1], [2], [2]])

    weights = _compute_equal_stratum_sample_weights(covariates)

    assert np.isclose(weights.mean(), 1.0)
    masses = []
    for value in [0, 1, 2]:
        masses.append(float(weights[covariates[:, 0] == value].sum()))
    assert np.allclose(masses, masses[0])


def test_demean_by_covariate_strata_zeroes_within_stratum_means() -> None:
    activations = np.array(
        [
            [1.0, 4.0],
            [3.0, 8.0],
            [10.0, 1.0],
            [14.0, 5.0],
        ]
    )
    target = np.array([0.0, 2.0, 10.0, 14.0])
    covariates = np.array([[0], [0], [1], [1]])

    Z_resid, y_resid, counts = _demean_by_covariate_strata(
        activations, target, covariates
    )

    assert counts == {(0,): 2, (1,): 2}
    for value in [0, 1]:
        mask = covariates[:, 0] == value
        assert np.allclose(Z_resid[mask].mean(axis=0), 0.0)
        assert np.isclose(y_resid[mask].mean(), 0.0)


def test_demeaned_reweighted_lasso_is_deterministic_and_selects_signal() -> None:
    rng = np.random.default_rng(7)
    n_per_stratum = 40
    covariates = np.array([[0]] * n_per_stratum + [[1]] * n_per_stratum)
    signal = np.concatenate(
        [
            rng.normal(size=n_per_stratum),
            rng.normal(size=n_per_stratum),
        ]
    )
    noise = rng.normal(size=(2 * n_per_stratum, 3))
    activations = np.column_stack([signal, noise])
    stratum_offset = np.where(covariates[:, 0] == 0, -3.0, 3.0)
    target = stratum_offset + 2.5 * signal + rng.normal(scale=0.05, size=2 * n_per_stratum)

    selected_a, coefs_a, info_a = select_neurons_demeaned_reweighted_lasso(
        activations=activations,
        target=target,
        n_select=1,
        covariates=covariates,
        return_info=True,
    )
    selected_b, coefs_b, info_b = select_neurons_demeaned_reweighted_lasso(
        activations=activations,
        target=target,
        n_select=1,
        covariates=covariates,
        return_info=True,
    )

    assert selected_a == selected_b == [0]
    assert np.allclose(coefs_a, coefs_b)
    assert info_a["n_nonzero"] == info_b["n_nonzero"]
    assert np.isclose(info_a["sample_weight_mean"], 1.0)


def test_kept_relative_indices_map_to_global_neuron_ids() -> None:
    kept_idx = np.array([3, 7, 11, 19])
    selected_rel = [2, 0]

    selected_global = kept_idx[np.asarray(selected_rel, dtype=int)].tolist()

    assert selected_global == [11, 3]
