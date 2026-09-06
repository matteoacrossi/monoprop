# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T3 and T4a acceptance criteria from PLAN_lsh_pauli_prop.md section 6.

T4a's acceptance criterion is written against ``ref_x100_perqubit_diff.csv``, but that file
turned out not to hold 120 independent per-qubit values -- see ``observable.py``'s module
docstring. It is still useful as a *site-level* regression fixture (each pair of columns
``(2r, 2r+1)`` duplicates one site value) and as the source of the scalar ``n_f(t)`` targets;
the true per-site ground truth used here is ``lsh_data/data_x100.h5``'s ``PP_stagg`` array,
which sums to the same ``n_f(t)`` values exactly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import pytest

from benches.hadron.propagate import n_f_at_layer, run
from benches.hadron.qasm_frontend import split_state_prep


def test_csv_site_columns_are_nearly_duplicated_pairs() -> None:
    """Document the discovery driving this module: CSV column pairs (2r, 2r+1) nearly coincide.

    Close but not exact -- the gap grows with the step (up to ~8e-4 by step 20), consistent
    with each column having been produced by a *slightly* different numerical path (rather than
    literally copy-pasted) for what section 3's derivation shows is the same site-level value.
    """
    path = Path(__file__).parent / "ref_x100_perqubit_diff.csv"
    with path.open() as f:
        max_gap = max(
            abs(float(row[f"q{2 * r}"]) - float(row[f"q{2 * r + 1}"]))
            for row in csv.DictReader(f)
            for r in range(60)
        )
    assert max_gap < 1e-3


def test_t0_state_prep_differs_only_at_the_meson_site(circuits_dir: Path) -> None:
    scv_prep, _ = split_state_prep(circuits_dir / "x_100_SCV.qasm")
    meson_prep, _ = split_state_prep(circuits_dir / "x_100_meson.qasm")
    assert set(scv_prep) - set(meson_prep) == {59}
    assert set(meson_prep) - set(scv_prep) == {60}


# (cutoff, lower_atol) per layer count: layer 1 is exact (only ~3 terms per wire); layers 2-3
# need a real budget -- an unbounded cutoff at 3 layers exhausts memory (millions of terms per
# wire x 120 wires). These are deliberately modest, not tuned for accuracy -- see T5 for a real
# tolerance sweep.
_BUDGET = {1: (1000, None), 2: (20, 1e-5), 3: (12, 1e-4)}
# Observed per-site error against PP_stagg stops shrinking well before this bound as the budget
# tightens further (see the PR description), so it is PP_stagg's own truncation, not this
# computation's -- PP_CPU (which PP_stagg equals) also uses "a cap on retained terms, calibrated
# by extrapolation from small systems" per the plan's fixtures section, not an exact result.
_MAX_SITE_ERROR = {1: 1e-6, 2: 1e-3, 3: 5e-3}


def test_conserved_charge_is_exact_when_nothing_is_truncated(
    circuits_dir: Path,
) -> None:
    """``60 - sum_r [<Z_i(r)> + <Z_o(r)>]/2`` is a conserved charge of these dynamics.

    Exact at an unbounded cutoff, so its drift under a real budget is a truncation-error
    estimate that needs no reference data at all -- see
    [propagate.Run.charge_drift][benches.hadron.propagate.Run.charge_drift]. Individual Pauli
    rotations in the reduced circuit do *not* conserve it (a lone ``XX`` anticommutes with
    ``Z_1 + Z_2``); it is conserved once a whole layer is applied, which is exactly where the
    channel and the measurement sit.
    """
    outcome = run(circuits_dir, max_layers=1, cutoff=1000)
    assert outcome.charge_scv == pytest.approx(60.0, abs=1e-12)
    assert outcome.charge_meson == pytest.approx(60.0, abs=1e-12)


@pytest.mark.parametrize("max_layers", [1, 2, 3])
def test_t4a_per_site_matches_h5_pp_stagg(
    circuits_dir: Path, h5_path: Path, max_layers: int
) -> None:
    """T4a: per-site agreement against the raw PP_CPU-equivalent reference (``PP_stagg``)."""
    cutoff, lower_atol = _BUDGET[max_layers]
    _, per_site = n_f_at_layer(
        circuits_dir, max_layers=max_layers, cutoff=cutoff, lower_atol=lower_atol
    )
    with h5py.File(h5_path, "r") as f:
        target = f["PP_stagg"][:, max_layers - 1]
    max_err = max(abs(mine - ref) for mine, ref in zip(per_site, target, strict=True))
    assert max_err < _MAX_SITE_ERROR[max_layers], (
        f"max per-site error {max_err} at {max_layers} layer(s)"
    )


@pytest.mark.parametrize(
    ("max_layers", "target_n_f"),
    [(1, 1.825336), (2, 1.347871), (3, 0.693895)],
)
def test_t4a_scalar_n_f_matches_pp_gpu(
    circuits_dir: Path, max_layers: int, target_n_f: float
) -> None:
    """T4a: n_f(t) against the published PP_GPU column, at PP_GPU's own tolerance.

    PP_GPU used a 1e-5 coefficient threshold (see the plan's fixtures section); the tolerance
    here is not this computation's own accuracy (see ``test_t4a_per_site_matches_h5_pp_stagg``)
    but slack for comparing against a *different* truncation scheme entirely -- see
    ``PLAN_lsh_pauli_prop.md`` T5 for a real tolerance-vs-cost sweep.
    """
    cutoff, lower_atol = _BUDGET[max_layers]
    n_f, _ = n_f_at_layer(
        circuits_dir, max_layers=max_layers, cutoff=cutoff, lower_atol=lower_atol
    )
    assert n_f == pytest.approx(target_n_f, abs=5e-3)
