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

"""Section 8's prerequisites: pin the Jordan-Wigner convention and the cutoff semantics.

The plan is explicit that no basis comparison is trustworthy until (i) ``cutoff`` is known to
mean the same thing in both propagators and (ii) the Jordan-Wigner convention is identical on
both sides. Both are checked here, the second against hand-computed images rather than a
round-trip, since the API has no Majorana-to-Pauli inverse to round-trip through.

The strongest check is that the two bases agree *exactly* untruncated: Jordan-Wigner is a
bijection on strings, so the branching trees must match term for term, and any disagreement is
a bug in the generator conversion, the phases, or the expectation evaluator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import monoprop
from benches.hadron import majorana
from benches.hadron.pauli_frontend import reduce_body
from benches.hadron.propagate import NUM_QUBITS, RZ_PER_LAYER
from benches.hadron.propagate import run as pauli_run
from benches.hadron.qasm_frontend import split_state_prep

_SMALL = 4


def _jw(pauli: monoprop.Pauli) -> dict[tuple[int, ...], complex]:
    operator = monoprop.PauliOperator({pauli: 1.0}, num_qubits=_SMALL)
    return {
        tuple(getattr(term, "indices", term)): coefficient
        for term, coefficient in operator.get_majorana_operator().terms.items()
    }


def test_jordan_wigner_images_match_the_hand_computed_convention() -> None:
    """Pin the convention: ``m_2j = (prod Z) X_j``, ``m_2j+1 = (prod Z) Y_j``.

    Under it ``m_2j m_2j+1 = i Z_j``, so ``Z_j = -i m_2j m_2j+1``; ``X_0 X_1 = -i m_1 m_2``
    (the Jordan-Wigner strings cancel, leaving length 2); and ``Z_0 Z_1 = -m_0 m_1 m_2 m_3``
    (length 4). That last pair is the whole point of section 8 -- equal Pauli weight, different
    Majorana length.
    """
    assert _jw(monoprop.Pauli("Z", (0,))) == {(0, 1): -1j}
    assert _jw(monoprop.Pauli("Z", (2,))) == {(4, 5): -1j}
    assert _jw(monoprop.Pauli("XX", (0, 1))) == {(1, 2): -1j}
    assert _jw(monoprop.Pauli("ZZ", (0, 1))) == {(0, 1, 2, 3): -1.0}


def test_hopping_and_density_terms_differ_in_length_but_not_weight() -> None:
    """The asymmetry section 8 rests on, stated as an assertion rather than prose."""
    hopping = monoprop.Pauli("XX", (0, 1))
    density = monoprop.Pauli("ZZ", (0, 1))
    assert len(hopping.qubits) == len(density.qubits) == 2
    assert len(next(iter(_jw(hopping)))) == 2
    assert len(next(iter(_jw(density)))) == 4


def test_quadratic_fraction_of_the_generators(circuits_dir: Path) -> None:
    """Section 8's first diagnostic, as a regression: two thirds of the layer is Gaussian.

    A quadratic generator preserves a monomial's Majorana length exactly, so these 360 gates
    cannot push a term over a length cutoff at all.
    """
    _, body = split_state_prep(circuits_dir / "x_100_SCV.qasm")
    reduced = reduce_body(
        body, num_qubits=NUM_QUBITS, rz_per_layer=RZ_PER_LAYER, max_layers=1
    )
    histogram = majorana.length_histogram(reduced.rotations, num_qubits=NUM_QUBITS)
    assert histogram == {2: 360, 4: 60, 6: 116}
    assert sum(histogram.values()) == RZ_PER_LAYER


def test_majorana_reproduces_pauli_exactly_when_untruncated(
    circuits_dir: Path,
) -> None:
    """Jordan-Wigner is a bijection, so untruncated the two bases must agree term for term."""
    pauli = pauli_run(circuits_dir, max_layers=1, cutoff=1000)
    mine = majorana.run(circuits_dir, max_layers=1)
    assert mine.n_f == pytest.approx(pauli.n_f, abs=1e-12)
    for got, expected in zip(mine.per_site, pauli.per_site, strict=True):
        assert got == pytest.approx(expected, abs=1e-12)
    # Same branching tree, hence the same retained term count -- not merely the same answer.
    assert mine.peak_terms == pauli.peak_terms


def _weight_length_support(label: str, qubits: tuple[int, ...]) -> tuple[int, int, int]:
    """``(Pauli weight, Majorana length, Majorana orbital support)`` of one Pauli string."""
    pauli = monoprop.Pauli(label, qubits)
    operator = monoprop.PauliOperator({pauli: 1.0}, num_qubits=_SMALL)
    (term,) = operator.get_majorana_operator().terms
    indices = tuple(getattr(term, "indices", term))
    return len(pauli.qubits), len(indices), len({index // 2 for index in indices})


def test_cutoff_numbers_are_not_comparable_across_bases() -> None:
    """Prerequisite (i): the two cutoffs measure different things, so compare term counts.

    Neither Majorana notion equals Pauli weight, so a section-8 comparison must be read as
    error against *cost*, never at equal cutoff numbers.

    The plan frames the asymmetry as diagonal-versus-hopping -- ``X_j X_{j+1}`` is Majorana
    length 2 where ``Z_j Z_{j+1}`` is length 4 -- and that holds for *adjacent* pairs. But the
    dominant effect on a 120-site chain is the Jordan-Wigner tail: an isolated ``X_2`` is Pauli
    weight 1 and Majorana length 5, because the string of ``Z``s back to the origin has to be
    paid for. The penalty scales with a term's span along the ordering, and only ``Z``-like
    letters escape it. That, rather than the diagonal/hopping split, is what makes the Majorana
    length cutoff lose here.
    """
    assert _weight_length_support("XX", (0, 1)) == (2, 2, 2)
    assert _weight_length_support("ZZ", (0, 1)) == (2, 4, 2)
    assert _weight_length_support("X", (0,)) == (1, 1, 1)
    assert _weight_length_support("X", (2,)) == (1, 5, 3)
    assert _weight_length_support("Z", (2,)) == (1, 2, 1)
    assert _weight_length_support("XX", (0, 2)) == (2, 4, 3)


def test_basis_expectation_of_paired_and_unpaired_monomials() -> None:
    """The evaluator, against hand-computed values for the images pinned above."""
    # Z_0 = -i m_0 m_1: +1 on |0>, -1 on |1>.
    z_0 = monoprop.MajoranaOperator({(0, 1): -1j}, _SMALL)
    assert majorana.basis_expectation(z_0, ()) == pytest.approx(1.0)
    assert majorana.basis_expectation(z_0, (0,)) == pytest.approx(-1.0)
    assert majorana.basis_expectation(z_0, (1,)) == pytest.approx(1.0)

    # An unpaired monomial is an X or Y somewhere, so it is off-diagonal.
    assert majorana.basis_expectation(
        monoprop.MajoranaOperator({(1, 2): -1j}, _SMALL), ()
    ) == pytest.approx(0.0)

    # Z_0 Z_1 = -m_0 m_1 m_2 m_3: even parity on the pair of sites.
    z_0_z_1 = monoprop.MajoranaOperator({(0, 1, 2, 3): -1.0}, _SMALL)
    assert majorana.basis_expectation(z_0_z_1, ()) == pytest.approx(1.0)
    assert majorana.basis_expectation(z_0_z_1, (0,)) == pytest.approx(-1.0)
    assert majorana.basis_expectation(z_0_z_1, (0, 1)) == pytest.approx(1.0)

    # The identity carries expectation 1 regardless of the state.
    assert majorana.basis_expectation(
        monoprop.MajoranaOperator({(): 1.0}, _SMALL), (0, 2)
    ) == pytest.approx(1.0)
