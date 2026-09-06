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

"""Reduce the Clifford+RZ hadron-dynamics circuits into pure Pauli-rotation generators.

T1 of PLAN_lsh_pauli_prop.md: ``monoprop.from_qiskit_circuit`` only accepts
``PauliEvolutionGate``-equivalent rotations (see ``PAULI_EVOLUTION_EQUIVALENT`` in
``monoprop.qiskit_conversion``), not raw ``h``/``cx``/``swap`` gates, and its own docstring
requires a multi-term generator's Pauli terms to commute -- which rules out feeding a Hadamard
in as a two-term generator (``X`` and ``Z`` anticommute). So instead of converting through
qiskit, every ``cx``-ladder / ``h``-conjugated ``rz`` block is reduced by hand: writing
``D_i`` for the product of every Clifford gate (``h``, ``cx``, ``x``, ``swap``) strictly before
the i-th ``rz(theta) q[k]``, that gate is exactly ``exp(-i theta/2 * P_i)`` for the *single*
Pauli string ``P_i = D_i^-1 Z_k D_i`` (Heisenberg-picture conjugation -- getting this backwards,
``D_i Z_k D_i^-1``, silently produces a plausible-looking but wrong reduction, since it's only
the local frame relative to the *end* of the accumulated Cliffords rather than the start).
Composing the whole body this way pulls every Clifford gate rightward past the ``rz``s it
precedes, so the entire body equals ``D_final * R_M' * ... * R_1'`` for the reduced rotations
``R_i'`` and ``D_final`` (every Clifford gate in the body, composed together). The desired
*output* observable must be reduced through ``D_final`` the same way before it is handed to
monoprop -- see [reduce_output_pauli][].

Clifford gates map Pauli strings to Pauli strings exactly (no branching, no approximation), so
this reduction is exact; only the qiskit ``Clifford`` tableau is used as the (well-tested)
conjugation engine -- the resulting monoprop [Circuit][monoprop.circuit.Circuit] holds only
Pauli-rotation ExpGates.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.circuit.library import CXGate, HGate, SwapGate, XGate
from qiskit.quantum_info import Clifford
from qiskit.quantum_info import Pauli as _QiskitPauli

import monoprop
from benches.hadron.qasm_frontend import gate_name, gate_qubits

_CLIFFORD_GATES = {"h": HGate(), "cx": CXGate(), "x": XGate(), "swap": SwapGate()}
_SWAP_AS_TWO_QUBIT_GATES = 3


@dataclass(frozen=True, slots=True)
class PauliRotation:
    """One ``rz`` gate, reduced to the Pauli string it exponentiates.

    Attributes:
        pauli: The generator, in the qubit frame of the circuit's start (t=0).
        angle: The raw ``rz`` rotation angle from the QASM file; the sign from reducing through
            the Clifford frame is folded into [pauli][]'s coefficient, not here.
        sign: The reduction's sign (+1.0 or -1.0); see [pauli][].
    """

    pauli: monoprop.Pauli
    angle: float
    sign: float


@dataclass(frozen=True, slots=True)
class ReducedBody:
    """The result of reducing a (prefix of a) Trotter body.

    Attributes:
        rotations: One [PauliRotation][] per ``rz`` gate consumed, in circuit order.
        final_clifford: The accumulated Clifford (``D_final``) of every ``h``/``cx``/``x``/
            ``swap`` gate consumed. Reduce an output-time observable through it with
            [reduce_output_pauli][] before backpropagating through [rotations][].
        layer_cliffords: The Clifford accumulated up to the end of each layer -- entry ``l`` is
            every Clifford gate strictly before that layer's last ``rz``. Conjugating by it maps
            a term held at that boundary back to the physical frame (see [noise][]). The last
            entry equals [final_clifford][], since reduction stops at the last consumed ``rz``.
        layer_two_qubit_gates: Two-qubit gates in each layer, counting a ``swap`` as three, for
            sizing a per-layer error rate.
    """

    rotations: list[PauliRotation]
    final_clifford: Clifford
    layer_cliffords: list[Clifford]
    layer_two_qubit_gates: list[int]


def _z_at(qubit: int, num_qubits: int) -> _QiskitPauli:
    """A qiskit ``Pauli`` for a lone ``Z`` on ``qubit`` (qiskit labels are qubit-119..0, left to right)."""
    letters = ["I"] * num_qubits
    letters[num_qubits - 1 - qubit] = "Z"
    return _QiskitPauli("".join(letters))


def _to_monoprop_pauli(label: str, num_qubits: int) -> tuple[monoprop.Pauli, float]:
    """Convert a qiskit Pauli label (e.g. ``"-IXZI"``) to a monoprop [Pauli][] and a real sign.

    Conjugating a Hermitian operator (``Z``) by a unitary Clifford stays Hermitian, so the
    phase is always ``+1`` or ``-1`` here -- never ``+-i``.
    """
    sign = -1.0 if label[0] == "-" else 1.0
    body = label[1:] if label[0] in "+-" else label
    letters = []
    qubits = []
    for index, letter in enumerate(body):
        if letter != "I":
            letters.append(letter)
            qubits.append(num_qubits - 1 - index)
    return monoprop.Pauli("".join(letters), tuple(qubits)), sign


def reduce_output_pauli(
    final_clifford: Clifford, qubit: int, num_qubits: int
) -> tuple[monoprop.Pauli, float]:
    """Reduce an output-time ``Z_qubit`` observable through a [ReducedBody.final_clifford][].

    Args:
        final_clifford: The Clifford a [ReducedBody][] was reduced through.
        qubit: The output-time physical qubit the desired observable is ``Z`` on.
        num_qubits: Circuit width.

    Returns:
        The monoprop [Pauli][] generator and its real sign, exactly like a [PauliRotation][]'s
        ``(pauli, sign)`` -- pass ``{pauli: sign}`` as a one-term ``PauliOperator`` to
        [monoprop.PauliPropagator][] as ``initial_operator``.
    """
    frame = _z_at(qubit, num_qubits).evolve(final_clifford, frame="h")
    return _to_monoprop_pauli(frame.to_label(), num_qubits)


def reduce_body(
    body_lines: list[str],
    *,
    num_qubits: int = 120,
    rz_per_layer: int = 536,
    max_layers: int | None = None,
) -> ReducedBody:
    """Reduce a Clifford+RZ gate stream into pure Pauli rotations plus the leftover Clifford.

    Args:
        body_lines: Raw QASM gate lines, e.g. the second element of
            [qasm_frontend.split_state_prep][benches.hadron.qasm_frontend.split_state_prep]'s
            result. Must include ``swap``.
        num_qubits: Circuit width.
        rz_per_layer: Number of ``rz`` gates per Trotter layer, to know when ``max_layers`` is
            reached.
        max_layers: Stop after this many layers instead of consuming the whole body. ``None``
            processes every layer.

    Returns:
        The reduced rotations and the leftover Clifford transform.
    """
    cliff = Clifford.from_label("I" * num_qubits)
    rotations: list[PauliRotation] = []
    layer_cliffords: list[Clifford] = []
    layer_two_qubit_gates: list[int] = []
    rz_count_in_layer = 0
    two_qubit_in_layer = 0
    layers_done = 0
    for line in body_lines:
        if max_layers is not None and layers_done >= max_layers:
            break
        name = gate_name(line)
        qubits = gate_qubits(line)
        if name == "rz":
            frame = _z_at(qubits[0], num_qubits).evolve(cliff, frame="h")
            pauli, sign = _to_monoprop_pauli(frame.to_label(), num_qubits)
            angle = float(line[line.index("(") + 1 : line.index(")")])
            rotations.append(PauliRotation(pauli, angle, sign))
            rz_count_in_layer += 1
            if rz_count_in_layer % rz_per_layer == 0:
                rz_count_in_layer = 0
                layers_done += 1
                layer_cliffords.append(cliff)
                layer_two_qubit_gates.append(two_qubit_in_layer)
                two_qubit_in_layer = 0
            continue
        if name == "cx":
            two_qubit_in_layer += 1
        elif name == "swap":
            two_qubit_in_layer += _SWAP_AS_TWO_QUBIT_GATES
        cliff = cliff.compose(_CLIFFORD_GATES[name], qargs=list(qubits))
    return ReducedBody(
        rotations=rotations,
        final_clifford=cliff,
        layer_cliffords=layer_cliffords,
        layer_two_qubit_gates=layer_two_qubit_gates,
    )


def build_circuit(
    rotations: list[PauliRotation], *, num_qubits: int, initial_state: tuple[int, ...]
) -> monoprop.Circuit:
    """Build a monoprop [Circuit][monoprop.circuit.Circuit] from reduced Pauli rotations.

    Follows [monoprop.qiskit_conversion][]'s convention: each ``rz(theta)`` becomes an ExpGate
    with generator ``-0.5 * sign * pauli`` driven by the raw angle ``theta``, so ExpGate's
    ``exp(+i theta H)`` reproduces qiskit's ``RZ(theta) = exp(-i theta Z / 2)`` pulled back
    through the Clifford frame.

    Args:
        rotations: The reduced rotations, in circuit order.
        num_qubits: Circuit width.
        initial_state: Physical qubit indices occupied at t=0 (the state-prep ``x`` gates).

    Returns:
        The equivalent monoprop circuit.
    """
    gates = tuple(
        monoprop.ExpGate(
            monoprop.PauliOperator({r.pauli: -0.5 * r.sign}, num_qubits=num_qubits)
        )
        for r in rotations
    )
    parameters = tuple(r.angle for r in rotations)
    return monoprop.Circuit(
        gates=gates,
        system_size=num_qubits,
        parameters=parameters,
        initial_state=initial_state,
    )
