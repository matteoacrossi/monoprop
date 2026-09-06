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

"""Driver tying qasm_frontend, pauli_frontend and observable together (T4 of the plan).

Propagates the differential ``n_f(t)`` observable through the first ``max_layers`` Trotter
layers of the SU(2) LSH hadron-dynamics circuit, using one [PauliPropagator][monoprop.pauli_propagator.PauliPropagator]
run per output wire (shared between the SCV and meson evaluations, per section 3 of the plan --
back-propagate once, evaluate against both product states).
"""

from __future__ import annotations

from pathlib import Path

import monoprop
from benches.hadron.observable import occupation, site_diff
from benches.hadron.pauli_frontend import (
    ReducedBody,
    build_circuit,
    reduce_body,
    reduce_output_pauli,
)
from benches.hadron.qasm_frontend import split_state_prep

NUM_QUBITS = 120
RZ_PER_LAYER = 536


def per_wire_occupations(
    circuit: monoprop.Circuit,
    reduced: ReducedBody,
    scv_occupied: set[int],
    meson_occupied: set[int],
    *,
    cutoff: int,
    lower_atol: float | None = None,
) -> tuple[list[float], list[float]]:
    """SCV and meson occupation numbers for every output wire.

    Each wire is propagated once; both occupation numbers are read off the same
    [evolved_operator][monoprop.pauli_propagator.PauliPropagator.evolved_operator].

    Args:
        circuit: The reduced (pure Pauli-rotation) circuit, from [pauli_frontend.build_circuit][].
        reduced: The [ReducedBody][benches.hadron.pauli_frontend.ReducedBody] that produced
            ``circuit``, for its ``final_clifford``.
        scv_occupied: Physical qubits occupied in the SCV run.
        meson_occupied: Physical qubits occupied in the meson run.
        cutoff: Pauli-weight cutoff, forwarded to
            [PauliPropagator][monoprop.pauli_propagator.PauliPropagator].
        lower_atol: Coefficient-magnitude cutoff, forwarded to
            [PauliPropagator][monoprop.pauli_propagator.PauliPropagator].

    Returns:
        ``(occupation_scv, occupation_meson)``, each indexed by output wire ``0..119``.
    """
    occupation_scv = [0.0] * NUM_QUBITS
    occupation_meson = [0.0] * NUM_QUBITS
    for wire in range(NUM_QUBITS):
        pauli, sign = reduce_output_pauli(reduced.final_clifford, wire, NUM_QUBITS)
        initial_operator = monoprop.PauliOperator({pauli: sign}, num_qubits=NUM_QUBITS)
        prop = monoprop.PauliPropagator.from_circuit(
            circuit, initial_operator, cutoff=cutoff, lower_atol=lower_atol
        )
        evolved = prop.evolved_operator()
        occupation_scv[wire] = occupation(evolved, scv_occupied)
        occupation_meson[wire] = occupation(evolved, meson_occupied)
    return occupation_scv, occupation_meson


def n_f_at_layer(
    circuits_dir: str | Path,
    *,
    max_layers: int,
    cutoff: int = 1000,
    lower_atol: float | None = None,
) -> tuple[float, list[float]]:
    """Compute ``n_f(t)`` after ``max_layers`` Trotter layers.

    Args:
        circuits_dir: Directory holding ``x_100_SCV.qasm`` and ``x_100_meson.qasm``.
        max_layers: Number of Trotter layers to include.
        cutoff: Pauli-weight cutoff, forwarded to
            [PauliPropagator][monoprop.pauli_propagator.PauliPropagator].
        lower_atol: Coefficient-magnitude cutoff, forwarded to
            [PauliPropagator][monoprop.pauli_propagator.PauliPropagator].

    Returns:
        ``(n_f(t), per_site_diff)``: the scalar observable and each of the 60 sites'
        contribution to it, in site order.
    """
    circuits_dir = Path(circuits_dir)
    scv_prep, body = split_state_prep(circuits_dir / "x_100_SCV.qasm")
    meson_prep, _ = split_state_prep(circuits_dir / "x_100_meson.qasm")

    reduced = reduce_body(
        body, num_qubits=NUM_QUBITS, rz_per_layer=RZ_PER_LAYER, max_layers=max_layers
    )
    circuit = build_circuit(
        reduced.rotations, num_qubits=NUM_QUBITS, initial_state=scv_prep
    )

    occupation_scv, occupation_meson = per_wire_occupations(
        circuit,
        reduced,
        set(scv_prep),
        set(meson_prep),
        cutoff=cutoff,
        lower_atol=lower_atol,
    )
    per_site = [
        site_diff(occupation_meson, occupation_scv, r) for r in range(NUM_QUBITS // 2)
    ]
    return sum(per_site), per_site
