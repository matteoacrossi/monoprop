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

"""The same circuit in the Majorana basis (section 8 of PLAN_lsh_pauli_prop.md).

Jordan-Wigner is a bijection between Pauli strings and Majorana monomials, so *untruncated*
the two propagators produce identical branching trees term for term -- which is exactly why
this is worth building: any disagreement is a bug, and
``test_majorana.py`` asserts the agreement across all 120 wires.

What differs is the discard order, because weight differs. ``X_j X_{j+1}`` is Pauli weight 2 and
Majorana length 2, while ``Z_j Z_{j+1}`` is Pauli weight 2 but Majorana length 4: diagonal,
density-like terms are penalised under a Majorana length cutoff and hopping-like terms are
favoured. Note this only bites through a *weight* cutoff. Under a pure coefficient threshold
the two bases discard precisely the same terms, since a term's coefficient is basis-independent,
so no advantage is possible on that axis -- see the README.

The plan's first diagnostic ([length_histogram][]) is the reason to bother: 360 of this
circuit's 536 generators per layer are Majorana-*quadratic*. A quadratic generator is Gaussian,
and Gaussian conjugation maps each ``m_k`` to a length-1 combination of Majoranas, so it
preserves a monomial's length exactly. Two thirds of the gates therefore cannot push a term
over a length cutoff at all, which is a structural reason to expect an advantage rather than
merely a reordering argument.

There is no compression from "working in fermionic space": 60 sites = 120 modes = 240
Majoranas, and the qubit count already equals the mode count.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter

import monoprop
from benches.hadron.observable import site_diff
from benches.hadron.pauli_frontend import reduce_body, reduce_output_pauli
from benches.hadron.propagate import NUM_QUBITS, RZ_PER_LAYER, Run
from benches.hadron.qasm_frontend import split_state_prep


def _term_indices(term: object) -> tuple[int, ...]:
    """Majorana indices of a term, whether the engine hands back a ``Majorana`` or a tuple."""
    return tuple(getattr(term, "indices", term))  # type: ignore[arg-type]


def hermitian_generator(
    pauli: monoprop.Pauli, sign: float, num_qubits: int
) -> monoprop.MajoranaOperator:
    """Jordan-Wigner image of a reduced rotation's generator, ready for an ExpGate.

    Mirrors [pauli_frontend.build_circuit][benches.hadron.pauli_frontend.build_circuit]'s
    convention -- generator ``-0.5 * sign * pauli`` driven by the raw angle -- and lets
    [PauliOperator.get_majorana_operator][monoprop.pauli.PauliOperator.get_majorana_operator]
    supply the Jordan-Wigner phases, which already match ExpGate's Hermitian Majorana
    convention (imaginary coefficient for a weight-2 monomial, real for weight-4).

    Args:
        pauli: The reduced generator's Pauli string.
        sign: The reduction's sign.
        num_qubits: Circuit width.

    Returns:
        The Majorana generator.
    """
    return monoprop.PauliOperator(
        {pauli: -0.5 * sign}, num_qubits=num_qubits
    ).get_majorana_operator()


def build_circuit(
    rotations: list, *, num_qubits: int, initial_state: tuple[int, ...]
) -> monoprop.Circuit:
    """Build the Majorana-family circuit from the same reduced rotations the Pauli path uses.

    Args:
        rotations: [PauliRotation][benches.hadron.pauli_frontend.PauliRotation] list, in
            circuit order.
        num_qubits: Circuit width.
        initial_state: Physical qubit indices occupied at t=0.

    Returns:
        A [Circuit][monoprop.circuit.Circuit] whose ``family`` is ``"majorana"``.
    """
    gates = tuple(
        monoprop.ExpGate(hermitian_generator(r.pauli, r.sign, num_qubits))
        for r in rotations
    )
    return monoprop.Circuit(
        gates=gates,
        system_size=num_qubits,
        parameters=tuple(r.angle for r in rotations),
        initial_state=initial_state,
    )


def basis_expectation(
    evolved: monoprop.MajoranaOperator, occupied: set[int] | tuple[int, ...]
) -> float:
    """Expectation of a Majorana operator against a computational basis product state.

    A monomial is diagonal only if its indices pair up as adjacent ``(2j, 2j+1)``, since
    ``m_{2j} m_{2j+1} = i Z_j`` while any unpaired index leaves an ``X`` or ``Y`` behind and so
    has zero expectation. The pairs come out already sorted, so no reordering sign appears and a
    monomial spanning sites ``S`` contributes ``i**|S| * (-1)**(occupied sites in S)``.

    The imaginary parts cancel for a Hermitian operator; the result is returned real, and
    ``test_majorana.py`` checks the residual imaginary part stays at round-off.

    Args:
        evolved: A backpropagated Majorana operator.
        occupied: The qubit indices set to |1>.

    Returns:
        The expectation value.
    """
    occupied_set = set(occupied)
    total = 0j
    for term, coefficient in evolved.terms.items():
        indices = _term_indices(term)
        if len(indices) % 2:
            continue
        sites = []
        for even, odd in zip(indices[::2], indices[1::2], strict=True):
            if even % 2 or odd != even + 1:
                break
            sites.append(even // 2)
        else:
            parity = -1 if sum(site in occupied_set for site in sites) % 2 else 1
            total += coefficient * (1j ** len(sites)) * parity
    return total.real


def length_histogram(rotations: list, *, num_qubits: int) -> dict[int, int]:
    """Majorana length of every reduced generator -- the plan's first section-8 diagnostic.

    Args:
        rotations: [PauliRotation][benches.hadron.pauli_frontend.PauliRotation] list.
        num_qubits: Circuit width.

    Returns:
        A ``length -> generator count`` mapping. Length 2 is the quadratic (Gaussian,
        length-preserving) fraction.
    """
    histogram: Counter[int] = Counter()
    for rotation in rotations:
        operator = monoprop.PauliOperator({rotation.pauli: 1.0}, num_qubits=num_qubits)
        (term,) = operator.get_majorana_operator().terms
        histogram[len(_term_indices(term))] += 1
    return dict(sorted(histogram.items()))


def run(
    circuits_dir: str | Path,
    *,
    max_layers: int,
    cutoff: int = 2 * NUM_QUBITS,
    lower_atol: float | None = None,
    cutoff_type: str = "length",
) -> Run:
    """Propagate ``n_f(t)`` in the Majorana basis.

    Deliberately a separate function rather than a mode of
    [propagate.run][benches.hadron.propagate.run]: the propagator class, the cutoff semantics
    and the expectation evaluator all differ, and there is no channel here (H3 is Pauli-side).

    Args:
        circuits_dir: Directory holding ``x_100_SCV.qasm`` and ``x_100_meson.qasm``.
        max_layers: Number of Trotter layers to include.
        cutoff: Read according to ``cutoff_type``; defaults to ``2 * NUM_QUBITS``, i.e.
            unbounded.
        lower_atol: Coefficient-magnitude cutoff.
        cutoff_type: ``"length"`` bounds the number of Majorana operators in a term -- the
            notion that differs from Pauli weight, and the point of the comparison.
            ``"support"`` bounds the modes touched, which *is* Pauli weight, so it should
            reproduce the Pauli path's truncation exactly.

    Returns:
        The [Run][benches.hadron.propagate.Run].
    """
    started = perf_counter()
    circuits_dir = Path(circuits_dir)
    scv_prep, body = split_state_prep(circuits_dir / "x_100_SCV.qasm")
    meson_prep, _ = split_state_prep(circuits_dir / "x_100_meson.qasm")

    reduced = reduce_body(
        body, num_qubits=NUM_QUBITS, rz_per_layer=RZ_PER_LAYER, max_layers=max_layers
    )
    circuit = build_circuit(
        reduced.rotations, num_qubits=NUM_QUBITS, initial_state=scv_prep
    )

    occupation_scv = [0.0] * NUM_QUBITS
    occupation_meson = [0.0] * NUM_QUBITS
    peak_terms = 0
    total_terms = 0
    scv_occupied = set(scv_prep)
    meson_occupied = set(meson_prep)

    for wire in range(NUM_QUBITS):
        pauli, sign = reduce_output_pauli(reduced.final_clifford, wire, NUM_QUBITS)
        observable = monoprop.PauliOperator(
            {pauli: sign}, num_qubits=NUM_QUBITS
        ).get_majorana_operator()
        propagator = monoprop.MajoranaPropagator.from_circuit(
            circuit,
            observable,
            cutoff=cutoff,
            cutoff_type=cutoff_type,
            lower_atol=lower_atol,
        )
        size = propagator.size()
        peak_terms = max(peak_terms, size)
        total_terms += size
        evolved = propagator.evolved_operator()
        occupation_scv[wire] = occupation_from(evolved, scv_occupied)
        occupation_meson[wire] = occupation_from(evolved, meson_occupied)

    per_site = [
        site_diff(occupation_meson, occupation_scv, r) for r in range(NUM_QUBITS // 2)
    ]
    return Run(
        n_f=sum(per_site),
        per_site=per_site,
        charge_scv=sum(occupation_scv),
        charge_meson=sum(occupation_meson),
        peak_terms=peak_terms,
        total_terms=total_terms,
        seconds=perf_counter() - started,
    )


def occupation_from(evolved: monoprop.MajoranaOperator, occupied: set[int]) -> float:
    """Occupation number ``(1 - <Z>) / 2`` from a Majorana-basis operator.

    The Majorana-basis twin of
    [observable.occupation][benches.hadron.observable.occupation].

    Args:
        evolved: A backpropagated Majorana operator for a single-qubit ``Z`` observable.
        occupied: The qubit indices set to |1>.

    Returns:
        The occupation number.
    """
    return (1 - basis_expectation(evolved, occupied)) / 2


__all__ = [
    "basis_expectation",
    "build_circuit",
    "hermitian_generator",
    "length_histogram",
    "occupation_from",
    "run",
]
