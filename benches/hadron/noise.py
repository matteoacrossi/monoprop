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

"""Weight-dependent depolarizing damping (H3 / T6 of PLAN_lsh_pauli_prop.md).

monoprop has no channel in its API, so T6's fallback applies: a depolarizing channel acting
independently on each qubit multiplies a Pauli term of weight ``w`` by ``(1-p)^w``, which is a
coefficient rescale of the backpropagated operator at each layer boundary. The channel is
unital and self-adjoint in the Pauli basis, so the same rescale is correct in the Heisenberg
picture, where it is applied *before* backpropagating the layer it follows.

Two things make this less trivial than it looks.

**The weight must be the physical one.** ``pauli_frontend`` pulls every Clifford gate rightward
past the rotations it precedes, so a term held between reduced layers lives in a rotated frame:
writing ``C_l`` for the Cliffords accumulated through layer ``l``, the physical operator is
``C_l P C_l^-1`` (Schrodinger conjugation, qiskit's ``frame="s"`` -- the opposite frame from the
reduction itself). Cliffords do not preserve Pauli weight, so damping the reduced-frame weight
would be a different channel entirely. [physical_weights][] therefore maps each term through
``C_l`` before weighing it.

The correction turns out to be mild for *this* circuit -- a layer's accumulated Clifford is
close to a relabelling, so the observable itself comes back at weight 1 in both frames, which is
the same reason ``observable.py``'s site mapping ended up being plain consecutive pairs -- but
that is a measured property of these files, not something to rely on. The frame is cheap to get
right, and getting it wrong is invisible: it still produces a smooth, monotonically decaying
curve.

Conjugating millions of terms through qiskit one at a time is far too slow, so only the images
of the ``2n`` single-qubit generators are taken from qiskit; a term's image is then the XOR of
its letters' images in the symplectic ``(x, z)`` bit representation, and its weight is
``popcount(x | z)``. Python's big integers make that a handful of machine words per term.

**Global depolarizing is not a test of H3.** A *global* depolarizing channel (damping every
non-identity Pauli by one factor, independent of weight) is Clifford-covariant and commutes
through the whole computation, so it predicts exactly ``n_f_noisy(t) = (1-q)^t n_f(t)`` -- which
is section 4's fitted ``F(t) = exp(-lambda t)`` and needs no simulation at all. It also yields
no cost saving, since a uniform rescale does not preferentially damp the high-weight terms that
drive the branching. H3's content is specific to the *weight-dependent* channel implemented
here: it damps exactly the terms that proliferate, and its predicted curve need not be a pure
exponential.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.quantum_info import Pauli as _QiskitPauli

import monoprop

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qiskit.quantum_info import Clifford

#: Per-two-qubit-gate depolarizing error rate, from section 4 of the plan.
DEFAULT_TWO_QUBIT_ERROR = 2e-3

#: A routing ``swap`` costs three ``cx``, per the plan's two-qubit gate accounting.
SWAP_AS_TWO_QUBIT_GATES = 3

#: ``(qubit, letter) -> (x_mask, z_mask)`` images of the single-qubit Paulis under a Clifford.
MaskTable = dict[tuple[int, str], tuple[int, int]]


def per_qubit_depolarizing(
    two_qubit_gates: int,
    *,
    num_qubits: int,
    two_qubit_error: float = DEFAULT_TWO_QUBIT_ERROR,
) -> float:
    """Per-qubit depolarizing probability for one layer, from that layer's gate count.

    A two-qubit gate error is charged to both of its qubits, so a layer of ``two_qubit_gates``
    gates touches the average qubit ``2 * two_qubit_gates / num_qubits`` times. This is a
    coarse-graining of per-gate noise onto layer boundaries, not a device model.

    Args:
        two_qubit_gates: Two-qubit gates in the layer, counting a ``swap`` as three.
        num_qubits: Circuit width.
        two_qubit_error: Depolarizing error per two-qubit gate.

    Returns:
        The per-qubit depolarizing probability ``p`` for one layer.
    """
    gates_per_qubit = 2 * two_qubit_gates / num_qubits
    return 1.0 - (1.0 - two_qubit_error) ** gates_per_qubit


def _label_to_masks(label: str, num_qubits: int) -> tuple[int, int]:
    """Symplectic ``(x, z)`` bit masks of a qiskit Pauli label (which reads qubit ``n-1`` first)."""
    body = label.lstrip("+-i")
    x = z = 0
    for index, letter in enumerate(body):
        qubit = num_qubits - 1 - index
        if letter in "XY":
            x |= 1 << qubit
        if letter in "ZY":
            z |= 1 << qubit
    return x, z


def pauli_image_masks(clifford: Clifford, num_qubits: int) -> MaskTable:
    """Images of every single-qubit Pauli under ``P -> C P C^-1``, as symplectic bit masks.

    Args:
        clifford: The Clifford to conjugate by.
        num_qubits: Circuit width.

    Returns:
        A ``(qubit, letter) -> (x_mask, z_mask)`` table covering ``X``, ``Y`` and ``Z`` on every
        qubit. Phases are dropped: only the support of the image matters for a weight.
    """
    images: MaskTable = {}
    for qubit in range(num_qubits):
        for letter in "XZ":
            letters = ["I"] * num_qubits
            letters[num_qubits - 1 - qubit] = letter
            image = _QiskitPauli("".join(letters)).evolve(clifford, frame="s")
            images[qubit, letter] = _label_to_masks(image.to_label(), num_qubits)
        x_x, x_z = images[qubit, "X"]
        z_x, z_z = images[qubit, "Z"]
        # Y ~ i X Z, and Pauli multiplication is addition of symplectic vectors.
        images[qubit, "Y"] = (x_x ^ z_x, x_z ^ z_z)
    return images


def physical_weights(
    operator: monoprop.PauliOperator, images: MaskTable
) -> dict[monoprop.Pauli, int]:
    """Physical-frame Pauli weight of every term of a reduced-frame operator.

    Args:
        operator: A backpropagated operator, in the reduced (rotated) frame.
        images: The frame's mask table, from [pauli_image_masks][].

    Returns:
        A ``term -> weight`` mapping.
    """
    weights: dict[monoprop.Pauli, int] = {}
    for term in operator.terms:
        x = z = 0
        for qubit, letter in zip(term.qubits, term.string, strict=True):
            image_x, image_z = images[qubit, letter]
            x ^= image_x
            z ^= image_z
        weights[term] = (x | z).bit_count()
    return weights


def damped(
    operator: monoprop.PauliOperator, probability: float, images: MaskTable
) -> monoprop.PauliOperator:
    """Apply one layer of per-qubit depolarizing damping to a backpropagated operator.

    Args:
        operator: The operator to damp, in the reduced (rotated) frame.
        probability: Per-qubit depolarizing probability ``p``; each term is scaled by
            ``(1-p)^w`` in its *physical* weight ``w``.
        images: The frame's mask table, from [pauli_image_masks][].

    Returns:
        The damped operator.
    """
    factors = _damping_factors(probability, operator.num_qubits)
    weights = physical_weights(operator, images)
    return monoprop.PauliOperator(
        {
            term: coefficient * factors[weights[term]]
            for term, coefficient in operator.terms.items()
        },
        num_qubits=operator.num_qubits,
    )


def dropped(operator: monoprop.PauliOperator, atol: float) -> monoprop.PauliOperator:
    """Discard terms below a coefficient threshold.

    Kept separate from [damped][] so that a noiseless control truncates a layer boundary on
    exactly the same terms a damped run does. Folding the two together would let the control
    carry sub-threshold terms across boundaries that the damped run drops, crediting the
    channel with a cost saving that is really just an asymmetric truncation.

    Args:
        operator: The operator to truncate.
        atol: Discard terms with ``|coefficient| < atol``.

    Returns:
        The truncated operator.
    """
    return monoprop.PauliOperator(
        {
            term: coefficient
            for term, coefficient in operator.terms.items()
            if abs(coefficient) >= atol
        },
        num_qubits=operator.num_qubits,
    )


def _damping_factors(probability: float, num_qubits: int) -> list[float]:
    """``(1-p)^w`` for every reachable weight ``w``."""
    factors = [1.0]
    for _ in range(num_qubits):
        factors.append(factors[-1] * (1.0 - probability))
    return factors


def weight_histogram(
    operator: monoprop.PauliOperator, images: MaskTable | None = None
) -> Mapping[int, int]:
    """Term count by Pauli weight, for reporting how the damping bites.

    Args:
        operator: The operator to profile.
        images: Frame mask table to report *physical* weights, from [pauli_image_masks][];
            ``None`` reports reduced-frame weights instead.

    Returns:
        A ``weight -> term count`` mapping.
    """
    histogram: dict[int, int] = {}
    if images is None:
        for term in operator.terms:
            histogram[len(term.qubits)] = histogram.get(len(term.qubits), 0) + 1
    else:
        for weight in physical_weights(operator, images).values():
            histogram[weight] = histogram.get(weight, 0) + 1
    return dict(sorted(histogram.items()))
