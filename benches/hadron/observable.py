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

"""The n_f observable and the LSH occupation-number formula (T3 of PLAN_lsh_pauli_prop.md).

The plan's own n_f formula (a linear, alternating-sign sum of raw ``<Z_j>`` differences) does
not hold up: it was never checked against real dynamics, only against the ``t=0`` product
states, where it happens to agree by coincidence (any formula linear in a *sum* of ``<Z>``
differences gives the same answer on a pure computational-basis state). The paper
(arXiv:2602.18080v3, "Mapping string-ends to qubits") instead defines the observable per
lattice site ``r`` (``r = 0..N-1``) from two *occupation numbers*, one qubit each:

    n_f(r) = n_i(r) + n_o(r)          if r is even
    n_f(r) = 2 - [n_i(r) + n_o(r)]    if r is odd
    n_f(t) = sum_r n_f(r, t)

with ``n_i(r) = (1 - <Z>) / 2`` for one of the site's qubits and ``n_o(r)`` likewise for the
other. Reconciling which *output* qubit is which required backpropagating through the circuit's
reduced Clifford (see ``pauli_frontend.reduce_output_pauli``) and matching against the raw
per-site data in ``lsh_data/data_x100.h5`` (``PP_Ni``, ``PP_No``, ``PP_scv``, ``PP_mid``,
``PP_stagg``) rather than the derived ``ref_x*_perqubit_diff.csv``, whose 120 columns turned out
to be each site's single differential value duplicated across both of that site's qubits, not
two independent measurements (every pair of columns ``(2r, 2r+1)`` is identical throughout the
file).

The reconciled rule below is validated exactly (to float precision) against every one of the 60
sites in ``data_x100.h5``'s SCV and meson step-1 columns: site ``r``'s two qubits are output
wires ``2r`` and ``2r+1`` -- no further permutation correction needed, since
``reduce_output_pauli`` already accounts for the circuit's Clifford gates before this indexing
is applied -- and the ``2 - x`` flip applies to *odd* ``r``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import monoprop


def basis_expectation(
    evolved: monoprop.PauliOperator, occupied: Iterable[int]
) -> float:
    """Expectation value of a backpropagated operator against a computational basis state.

    Any term with a non-``Z`` letter has zero expectation against a computational basis product
    state, so only the pure-``Z`` terms contribute.

    Args:
        evolved: A backpropagated operator, e.g. from
            [PauliPropagator.evolved_operator][monoprop.pauli_propagator.PauliPropagator].
        occupied: The qubit indices set to |1> (all others are |0>).

    Returns:
        The expectation value.
    """
    occupied_set = set(occupied)
    total = 0.0
    for term, coeff in evolved.terms.items():
        if any(letter != "Z" for letter in term.string):
            continue
        sign = 1
        for qubit in term.qubits:
            if qubit in occupied_set:
                sign = -sign
        total += coeff * sign
    return total


def occupation(evolved: monoprop.PauliOperator, occupied: Iterable[int]) -> float:
    """Occupation number ``(1 - <Z>) / 2`` of a backpropagated single-qubit-``Z``-rooted operator.

    Args:
        evolved: A backpropagated operator for a single-qubit ``Z`` observable, e.g. from
            [pauli_frontend.reduce_output_pauli][benches.hadron.pauli_frontend.reduce_output_pauli]
            followed by [PauliPropagator.evolved_operator][monoprop.pauli_propagator.PauliPropagator].
        occupied: The qubit indices set to |1> (all others are |0>).

    Returns:
        The occupation number, in ``[0, 1]``.
    """
    return (1 - basis_expectation(evolved, occupied)) / 2


def site_diff(
    occupation_meson: Sequence[float], occupation_scv: Sequence[float], r: int
) -> float:
    """The site-level differential ``n_f(r, meson) - n_f(r, SCV)``.

    Args:
        occupation_meson: Per-output-wire occupation numbers (index 0..119) for the meson run.
        occupation_scv: Per-output-wire occupation numbers (index 0..119) for the SCV run.
        r: Site index, ``0..59``.

    Returns:
        The site's contribution to ``n_f(t)``.
    """
    a, b = 2 * r, 2 * r + 1
    diff = (occupation_meson[a] + occupation_meson[b]) - (
        occupation_scv[a] + occupation_scv[b]
    )
    return -diff if r % 2 == 1 else diff


def n_f(
    occupation_meson: Sequence[float],
    occupation_scv: Sequence[float],
    num_sites: int = 60,
) -> float:
    """The published differential observable, summed over every site's contribution.

    Args:
        occupation_meson: Per-output-wire occupation numbers (length ``2 * num_sites``) for the
            meson run.
        occupation_scv: Per-output-wire occupation numbers (length ``2 * num_sites``) for the SCV
            run.
        num_sites: Number of lattice sites (60 for the published instance).

    Returns:
        ``n_f(t)``.
    """
    return sum(site_diff(occupation_meson, occupation_scv, r) for r in range(num_sites))
