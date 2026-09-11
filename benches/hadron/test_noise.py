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

"""H3/T6's depolarizing damping: the two things that could silently be wrong.

The bit-mask weight shortcut and the frame it conjugates in are both invisible in the output --
a wrong frame or a mis-XORed mask still yields plausible, monotonically decaying numbers. Both
are pinned against qiskit here, and the layer-by-layer propagation path is pinned against the
one-shot path it has to agree with when the channel is switched off.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Clifford,
    Operator,
    Pauli,
    random_clifford,
    random_pauli,
)

import monoprop
from benches.hadron import noise
from benches.hadron.observable import basis_expectation
from benches.hadron.pauli_frontend import (
    build_circuit,
    reduce_body,
    reduce_output_pauli,
)
from benches.hadron.propagate import _propagate_wire, run
from benches.hadron.qasm_frontend import gate_name, gate_qubits

_NUM_QUBITS = 6


def _monoprop_pauli(label: str, num_qubits: int) -> monoprop.Pauli:
    body = label.lstrip("+-i")
    letters = []
    qubits = []
    for index, letter in enumerate(body):
        if letter != "I":
            letters.append(letter)
            qubits.append(num_qubits - 1 - index)
    return monoprop.Pauli("".join(letters), tuple(qubits))


def test_physical_weights_match_qiskit_conjugation() -> None:
    """The mask-XOR shortcut must agree with conjugating each term through qiskit directly."""
    clifford = random_clifford(_NUM_QUBITS, seed=11)
    images = noise.pauli_image_masks(clifford, _NUM_QUBITS)
    rng = np.random.default_rng(7)
    for _ in range(50):
        label = random_pauli(_NUM_QUBITS, seed=int(rng.integers(1 << 30))).to_label()
        term = _monoprop_pauli(label, _NUM_QUBITS)
        operator = monoprop.PauliOperator({term: 1.0}, num_qubits=_NUM_QUBITS)
        image = Pauli(label).evolve(clifford, frame="s")
        expected = sum(1 for letter in image.to_label().lstrip("+-i") if letter != "I")
        assert noise.physical_weights(operator, images)[term] == expected


def test_damping_uses_the_schrodinger_frame() -> None:
    """Guard the frame choice: ``frame="h"`` is the reduction's, and is wrong for the weight.

    Both frames give equal weights whenever the Clifford happens to be weight-preserving, so
    the assertion is that *some* term separates them -- otherwise this test proves nothing.
    """
    clifford = random_clifford(_NUM_QUBITS, seed=5)
    images = noise.pauli_image_masks(clifford, _NUM_QUBITS)
    rng = np.random.default_rng(3)
    separated = False
    for _ in range(50):
        label = random_pauli(_NUM_QUBITS, seed=int(rng.integers(1 << 30))).to_label()
        term = _monoprop_pauli(label, _NUM_QUBITS)
        operator = monoprop.PauliOperator({term: 1.0}, num_qubits=_NUM_QUBITS)
        wrong = Pauli(label).evolve(clifford, frame="h")
        wrong_weight = sum(
            1 for letter in wrong.to_label().lstrip("+-i") if letter != "I"
        )
        separated |= noise.physical_weights(operator, images)[term] != wrong_weight
    assert separated


def test_zero_probability_damping_is_the_identity() -> None:
    clifford = random_clifford(_NUM_QUBITS, seed=2)
    images = noise.pauli_image_masks(clifford, _NUM_QUBITS)
    operator = monoprop.PauliOperator(
        {monoprop.Pauli("XY", (0, 3)): 0.25, monoprop.Pauli("Z", (5,)): -1.5},
        num_qubits=_NUM_QUBITS,
    )
    assert noise.damped(operator, 0.0, images).terms == operator.terms


def test_damping_scales_by_physical_weight() -> None:
    """A weight-``w`` term must pick up exactly ``(1-p)^w``.

    Uses the identity frame to isolate the scaling law; which frame supplies ``w`` is what
    [test_physical_weights_match_qiskit_conjugation][] covers.
    """
    identity = Clifford.from_label("I" * _NUM_QUBITS)
    images = noise.pauli_image_masks(identity, _NUM_QUBITS)
    probability = 0.1
    terms = {
        monoprop.Pauli("Z", (0,)): 1.0,
        monoprop.Pauli("XY", (1, 2)): 1.0,
        monoprop.Pauli("XYZ", (3, 4, 5)): 1.0,
    }
    damped = noise.damped(
        monoprop.PauliOperator(terms, num_qubits=_NUM_QUBITS), probability, images
    )
    for term in terms:
        expected = (1.0 - probability) ** len(term.qubits)
        assert damped.terms[term] == pytest.approx(expected)


def test_per_qubit_depolarizing_charges_both_qubits() -> None:
    p = noise.per_qubit_depolarizing(60, num_qubits=120, two_qubit_error=1e-3)
    # 60 gates over 120 qubits touches the average qubit once.
    assert p == pytest.approx(1e-3, rel=1e-9)


def test_layered_path_with_no_noise_matches_the_one_shot_path(
    circuits_dir: Path,
) -> None:
    """The control for every H3 comparison: chunking by layer must not change the answer.

    The two paths evaluate differently, so they agree to rounding rather than to the bit: the
    one-shot path contracts in the engine, while the layered path has to materialise the operator
    at each boundary (that is where the channel is applied) and sum it in Python. Measured against
    ``PP_stagg``, which is exact at one layer, the engine contraction lands within ``1.1e-16`` and
    the Python summation within ``7.8e-13``, so the gap between them is the latter's accumulated
    rounding, not a disagreement about the physics.
    """
    one_shot = run(circuits_dir, max_layers=1, cutoff=1000)
    layered = run(circuits_dir, max_layers=1, cutoff=1000, two_qubit_error=0.0)
    assert layered.n_f == pytest.approx(one_shot.n_f, abs=1e-11)
    for mine, reference in zip(layered.per_site, one_shot.per_site, strict=True):
        assert mine == pytest.approx(reference, abs=1e-11)


_SMALL_QUBITS = 4
#: Two layers of two ``rz`` each, with Cliffords interleaved so the reduction is non-trivial
#: and no Clifford trails the last ``rz`` (which would put the final boundary mid-layer).
_SMALL_BODY = [
    "h q[0];",
    "cx q[0],q[1];",
    "rz(0.4) q[1];",
    "rz(0.7) q[2];",
    "cx q[1],q[2];",
    "swap q[0],q[3];",
    "h q[3];",
    "rz(0.3) q[0];",
    "rz(0.9) q[3];",
]


def _pauli_matrix(letter: str, qubit: int) -> np.ndarray:
    """Full-width matrix of a single-qubit Pauli, in qiskit's qubit ordering."""
    letters = ["I"] * _SMALL_QUBITS
    letters[_SMALL_QUBITS - 1 - qubit] = letter
    return Operator(Pauli("".join(letters))).data


def _depolarize(density: np.ndarray, probability: float) -> np.ndarray:
    """Independent single-qubit depolarizing on every qubit, as an explicit Kraus sum.

    Scales every non-identity Pauli by ``1 - probability``, so a weight-``w`` term picks up
    ``(1 - probability)**w`` -- the channel ``noise.damped`` implements by rescaling instead.
    """
    for qubit in range(_SMALL_QUBITS):
        kept = (1.0 - 0.75 * probability) * density
        for letter in "XYZ":
            pauli = _pauli_matrix(letter, qubit)
            kept = kept + (probability / 4.0) * (pauli @ density @ pauli)
        density = kept
    return density


def _brute_force_expectation(probability: float) -> float:
    """``<Z_0>`` after the small circuit with a depolarizing channel at each layer boundary."""
    segments = []
    circuit = QuantumCircuit(_SMALL_QUBITS)
    seen = 0
    for line in _SMALL_BODY:
        name = gate_name(line)
        qubits = gate_qubits(line)
        if name == "rz":
            circuit.rz(float(line[line.index("(") + 1 : line.index(")")]), qubits[0])
            seen += 1
            if seen % 2 == 0:
                segments.append(circuit)
                circuit = QuantumCircuit(_SMALL_QUBITS)
        elif name == "h":
            circuit.h(qubits[0])
        elif name == "x":
            circuit.x(qubits[0])
        elif name == "cx":
            circuit.cx(qubits[0], qubits[1])
        elif name == "swap":
            circuit.swap(qubits[0], qubits[1])
    density = np.zeros((2**_SMALL_QUBITS,) * 2, dtype=complex)
    density[0, 0] = 1.0
    for segment in segments:
        unitary = Operator(segment).data
        density = unitary @ density @ unitary.conj().T
        density = _depolarize(density, probability)
    return float(np.real(np.trace(density @ _pauli_matrix("Z", 0))))


@pytest.mark.parametrize("probability", [0.0, 0.05, 0.2])
def test_layered_damping_matches_a_brute_force_noisy_simulation(
    probability: float,
) -> None:
    """End-to-end: the rescale must equal an explicit Kraus depolarizing simulation.

    Placing the channel one layer off, or damping in the wrong frame, still yields a smooth
    decaying curve on the real circuit -- only a brute-force reference catches it.
    """
    reduced = reduce_body(_SMALL_BODY, num_qubits=_SMALL_QUBITS, rz_per_layer=2)
    layer_circuits = [
        build_circuit(
            reduced.rotations[start : start + 2],
            num_qubits=_SMALL_QUBITS,
            initial_state=(),
        )
        for start in (0, 2)
    ]
    damping = [
        (probability, noise.pauli_image_masks(clifford, _SMALL_QUBITS))
        for clifford in reduced.layer_cliffords
    ]
    pauli, sign = reduce_output_pauli(reduced.final_clifford, 0, _SMALL_QUBITS)
    evolved, _, _ = _propagate_wire(
        layer_circuits,
        damping,
        monoprop.PauliOperator({pauli: sign}, num_qubits=_SMALL_QUBITS),
        cutoff=_SMALL_QUBITS,
        lower_atol=None,
    )
    assert basis_expectation(evolved, ()) == pytest.approx(
        _brute_force_expectation(probability), abs=1e-10
    )


def test_damping_only_shrinks_the_observable(circuits_dir: Path) -> None:
    """A depolarizing channel cannot amplify: |n_f| must not grow when the channel is on."""
    noiseless = run(circuits_dir, max_layers=1, cutoff=1000, two_qubit_error=0.0)
    noisy = run(circuits_dir, max_layers=1, cutoff=1000, two_qubit_error=2e-3)
    assert abs(noisy.n_f) < abs(noiseless.n_f)
    assert noisy.peak_terms <= noiseless.peak_terms
