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

"""Driver tying qasm_frontend, pauli_frontend and observable together (T4 and T6 of the plan).

Propagates the differential ``n_f(t)`` observable through the first ``max_layers`` Trotter
layers of the SU(2) LSH hadron-dynamics circuit, using one [PauliPropagator][monoprop.pauli_propagator.PauliPropagator]
run per output wire (shared between the SCV and meson evaluations, per section 3 of the plan --
back-propagate once, evaluate against both product states).

Two propagation paths. The default consumes the whole reduced circuit in one propagator. Giving
[run][]'s ``two_qubit_error`` instead consumes one Trotter layer at a time so that H3's
depolarizing damping can be applied at each boundary (see [noise][]); passing ``0.0`` there is
the noiseless control for that path, and must agree with the default to truncation error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import monoprop
from benches.hadron import noise
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

#: Per layer boundary: the channel's ``(probability, frame mask table)``, or ``None`` for none.
_LayerDamping = tuple[float, noise.MaskTable] | None


@dataclass(frozen=True, slots=True)
class Run:
    """One propagation of ``n_f(t)``, with the cost it took.

    Attributes:
        n_f: The scalar differential observable.
        per_site: Each of the 60 sites' contribution to it, in site order.
        charge_scv: The conserved charge ``60 - sum_r [<Z_i(r)> + <Z_o(r)>] / 2`` of the SCV
            run, which is just the total occupation ``sum_j n(j)``. See [charge_drift][].
        charge_meson: The same for the meson run.
        peak_terms: Largest number of Pauli terms any single wire's graph held -- the memory
            bound.
        total_terms: Summed over every wire and layer -- the work proxy.
        seconds: Wall-clock time.
    """

    n_f: float
    per_site: list[float]
    charge_scv: float
    charge_meson: float
    peak_terms: int
    total_terms: int
    seconds: float

    @property
    def charge_drift(self) -> float:
        """Largest deviation of either run's charge from its exact value, ``NUM_QUBITS / 2``.

        A correctness invariant rather than an error estimate. The dynamics conserves the charge
        exactly, and so does the reduction, the site mapping and the channel, so a nonzero drift
        at shallow depth means something is broken. It is *not* a proxy for truncation error: at
        1-3 layers it stays at ``1e-14``-``1e-11`` under budgets that put ``1e-3`` of error on
        ``n_f``, because the contributions that cancel across wires carry equal-magnitude
        coefficients and a magnitude threshold keeps or drops both. Deeper it does drift, but
        non-monotonically in the budget -- see the README's table.

        The depolarizing channel leaves it alone too, which is not a coincidence worth trusting
        blindly: both initial states are exactly half filled, so ``sum_j <Z_j> = 0`` at ``t=0``,
        and a channel that scales every ``<Z_j>`` by ``1-p`` scales a zero sum to zero. That
        holds for *this* half-filled instance, not in general.
        """
        exact = NUM_QUBITS / 2
        return max(abs(self.charge_scv - exact), abs(self.charge_meson - exact))


def _layer_circuits(
    reduced: ReducedBody, *, initial_state: tuple[int, ...]
) -> list[monoprop.Circuit]:
    """Split the reduced rotations into one circuit per Trotter layer, in circuit order.

    Every ``rz`` reduces to exactly one rotation, so layer ``l`` owns rotations
    ``[RZ_PER_LAYER * l, RZ_PER_LAYER * (l + 1))``.
    """
    return [
        build_circuit(
            reduced.rotations[start : start + RZ_PER_LAYER],
            num_qubits=NUM_QUBITS,
            initial_state=initial_state,
        )
        for start in range(0, len(reduced.rotations), RZ_PER_LAYER)
    ]


def _damping_plan(reduced: ReducedBody, two_qubit_error: float) -> list[_LayerDamping]:
    """Per-layer ``(probability, frame mask table)``, or ``None`` where there is no damping.

    The frame tables are the expensive part (``2 * NUM_QUBITS`` qiskit conjugations each), so
    they are built once here and shared across all 120 wires.
    """
    if two_qubit_error == 0.0:
        return [None] * len(reduced.layer_cliffords)
    return [
        (
            noise.per_qubit_depolarizing(
                gates, num_qubits=NUM_QUBITS, two_qubit_error=two_qubit_error
            ),
            noise.pauli_image_masks(clifford, NUM_QUBITS),
        )
        for gates, clifford in zip(
            reduced.layer_two_qubit_gates, reduced.layer_cliffords, strict=True
        )
    ]


def _propagate_wire(
    layer_circuits: list[monoprop.Circuit],
    damping: list[_LayerDamping],
    initial_operator: monoprop.PauliOperator,
    *,
    cutoff: int,
    lower_atol: float | None,
) -> tuple[monoprop.PauliOperator, int, int]:
    """Backpropagate one wire layer by layer, damping at each boundary.

    The channel is applied *before* the layer it follows is consumed: backpropagation runs
    last-layer-first, and the channel after layer ``l`` is met before layer ``l`` itself. The
    ``lower_atol`` drop that follows it runs whether or not there is a channel, so a noiseless
    control truncates on the same rule -- see [noise.dropped][benches.hadron.noise.dropped].

    Returns:
        ``(evolved_operator, peak_terms, total_terms)``.
    """
    operator = initial_operator
    peak = 0
    total = 0
    for index in reversed(range(len(layer_circuits))):
        entry = damping[index]
        if entry is not None:
            probability, images = entry
            operator = noise.damped(operator, probability, images)
        if lower_atol is not None:
            operator = noise.dropped(operator, lower_atol)
        propagator = monoprop.PauliPropagator.from_circuit(
            layer_circuits[index], operator, cutoff=cutoff, lower_atol=lower_atol
        )
        size = propagator.size()
        peak = max(peak, size)
        total += size
        operator = propagator.evolved_operator()
    return operator, peak, total


def run(
    circuits_dir: str | Path,
    *,
    max_layers: int,
    cutoff: int = 1000,
    lower_atol: float | None = None,
    two_qubit_error: float | None = None,
) -> Run:
    """Propagate ``n_f(t)`` through ``max_layers`` Trotter layers.

    Args:
        circuits_dir: Directory holding ``x_100_SCV.qasm`` and ``x_100_meson.qasm``.
        max_layers: Number of Trotter layers to include.
        cutoff: Pauli-weight cutoff, forwarded to
            [PauliPropagator][monoprop.pauli_propagator.PauliPropagator].
        lower_atol: Coefficient-magnitude cutoff, forwarded to
            [PauliPropagator][monoprop.pauli_propagator.PauliPropagator], and the threshold for
            dropping damped terms at a layer boundary.
        two_qubit_error: Depolarizing error per two-qubit gate (H3/T6). ``None`` consumes the
            whole circuit in one propagator; a float switches to layer-by-layer propagation,
            with ``0.0`` as that path's noiseless control.

    Returns:
        The [Run][].
    """
    started = perf_counter()
    circuits_dir = Path(circuits_dir)
    scv_prep, body = split_state_prep(circuits_dir / "x_100_SCV.qasm")
    meson_prep, _ = split_state_prep(circuits_dir / "x_100_meson.qasm")

    reduced = reduce_body(
        body, num_qubits=NUM_QUBITS, rz_per_layer=RZ_PER_LAYER, max_layers=max_layers
    )

    occupation_scv = [0.0] * NUM_QUBITS
    occupation_meson = [0.0] * NUM_QUBITS
    peak_terms = 0
    total_terms = 0

    if two_qubit_error is None:
        circuit = build_circuit(
            reduced.rotations, num_qubits=NUM_QUBITS, initial_state=scv_prep
        )
        circuits = [circuit]
        damping: list[_LayerDamping] = [None]
    else:
        circuits = _layer_circuits(reduced, initial_state=scv_prep)
        damping = _damping_plan(reduced, two_qubit_error)

    scv_occupied = set(scv_prep)
    meson_occupied = set(meson_prep)
    for wire in range(NUM_QUBITS):
        pauli, sign = reduce_output_pauli(reduced.final_clifford, wire, NUM_QUBITS)
        initial_operator = monoprop.PauliOperator({pauli: sign}, num_qubits=NUM_QUBITS)
        evolved, peak, total = _propagate_wire(
            circuits, damping, initial_operator, cutoff=cutoff, lower_atol=lower_atol
        )
        occupation_scv[wire] = occupation(evolved, scv_occupied)
        occupation_meson[wire] = occupation(evolved, meson_occupied)
        peak_terms = max(peak_terms, peak)
        total_terms += total

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


def n_f_at_layer(
    circuits_dir: str | Path,
    *,
    max_layers: int,
    cutoff: int = 1000,
    lower_atol: float | None = None,
) -> tuple[float, list[float]]:
    """Compute ``n_f(t)`` after ``max_layers`` Trotter layers.

    A thin wrapper over [run][] keeping the noiseless two-tuple its callers expect.

    Args:
        circuits_dir: Directory holding ``x_100_SCV.qasm`` and ``x_100_meson.qasm``.
        max_layers: Number of Trotter layers to include.
        cutoff: Pauli-weight cutoff.
        lower_atol: Coefficient-magnitude cutoff.

    Returns:
        ``(n_f(t), per_site_diff)``: the scalar observable and each of the 60 sites'
        contribution to it, in site order.
    """
    outcome = run(
        circuits_dir, max_layers=max_layers, cutoff=cutoff, lower_atol=lower_atol
    )
    return outcome.n_f, outcome.per_site
