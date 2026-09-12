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

r"""Independent matrix-product-state check of ``n_f(t)``, via quimb.

Pauli propagation and this share only the circuit: the error here is bond-dimension truncation of
a state, there it is magnitude truncation of an operator. Neither approximation can imitate the
other, so agreement is evidence about the circuit rather than about either method's budget. That
is the point -- the published tensor-network column cannot serve as the cross-check, since it
evolves in continuous time and so carries no Trotter error, making it a different object.

The reduction does the hard part. ``D_final * R_M ... R_1`` leaves every ``R_i`` a Pauli rotation
of weight at most 2 spanning at most 4 consecutive qubits, so the routing swaps and ``cx`` ladders
that would otherwise force a swap network never reach the MPS: the gates are already local in the
natural site order. The final Clifford never touches the state either, because
[reduce_output_pauli][benches.hadron.pauli_frontend.reduce_output_pauli] hands back
``D_final^dag Z_w D_final`` as a Pauli string, and a Pauli string is a product of single-site
unitaries -- applying it cannot grow a bond. So the observable is evaluated exactly and the only
approximation in the whole calculation is the bond dimension.

Convergence is in ``max_bond`` here exactly as it is in ``lower_atol`` on the propagation side.
Run a ladder of bond dimensions and watch ``n_f`` settle; a single bond dimension says nothing.

Usage::

    uv run --no-sync python -m benches.hadron.mps --layers 20 --max-bond 128 256 512 800
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn

from benches.hadron.observable import site_diff
from benches.hadron.pauli_frontend import (
    PauliRotation,
    fuse_rotations,
    reduce_output_pauli,
)
from benches.hadron.propagate import (
    NUM_QUBITS,
    RZ_PER_LAYER,
    _reduce_body_cached,
)
from benches.hadron.qasm_frontend import split_state_prep

if TYPE_CHECKING:
    import monoprop

_HADRON_DIR = Path(__file__).parent
_CIRCUITS_DIR = (
    _HADRON_DIR
    / ".cache/qat/data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh"
)

_PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_matrix(pauli: monoprop.Pauli) -> tuple[np.ndarray, tuple[int, ...]]:
    """The dense matrix of a Pauli string and the ascending qubits it acts on."""
    qubits = sorted(zip(pauli.qubits, pauli.string, strict=True))
    matrix = np.array([[1.0 + 0j]])
    for _, letter in qubits:
        matrix = np.kron(matrix, _PAULI[letter])
    return matrix, tuple(qubit for qubit, _ in qubits)


def _rotation_gate(rotation: PauliRotation) -> tuple[np.ndarray, tuple[int, ...]]:
    """``exp(-i phi P / 2)`` for one reduced rotation.

    Mirrors [build_circuit][benches.hadron.pauli_frontend.build_circuit]'s convention: the gate is
    ``exp(+i theta H)`` for ``H = -0.5 * sign * P``, so the effective angle is ``sign * theta``.
    """
    matrix, qubits = _pauli_matrix(rotation.pauli)
    phi = rotation.sign * rotation.angle
    identity = np.eye(matrix.shape[0], dtype=complex)
    return np.cos(phi / 2) * identity - 1j * np.sin(phi / 2) * matrix, qubits


def _product_state(initial_state: tuple[int, ...]) -> qtn.MatrixProductState:
    """The computational-basis MPS with ``initial_state``'s qubits set to one."""
    bits = ["0"] * NUM_QUBITS
    for qubit in initial_state:
        bits[qubit] = "1"
    return qtn.MPS_computational_state("".join(bits))


def _apply_layer(
    psi: qtn.MatrixProductState,
    rotations: list[PauliRotation],
    *,
    max_bond: int,
    cutoff: float,
) -> None:
    """Apply one Trotter layer's rotations in place, truncating to ``max_bond``."""
    for rotation in rotations:
        gate, qubits = _rotation_gate(rotation)
        psi.gate_(
            gate,
            qubits,
            contract="swap+split" if len(qubits) > 1 else True,
            max_bond=max_bond,
            cutoff=cutoff,
        )


def _occupations(psi: qtn.MatrixProductState, final_clifford: object) -> list[float]:
    """``(1 - <Z_w>) / 2`` for every wire, with ``<Z_w>`` measured as a Pauli string on ``psi``.

    A Pauli string is a product of single-site unitaries, so applying it leaves the bond dimension
    alone and the expectation is exact at whatever bond the state was truncated to.
    """
    norm = psi.H @ psi
    occupations = []
    for wire in range(NUM_QUBITS):
        pauli, sign = reduce_output_pauli(final_clifford, wire, NUM_QUBITS)
        applied = psi.copy()
        for qubit, letter in zip(pauli.qubits, pauli.string, strict=True):
            applied.gate_(_PAULI[letter], (qubit,), contract=True)
        expectation = sign * ((psi.H @ applied) / norm).real
        occupations.append((1 - expectation) / 2)
    return occupations


def main() -> None:
    """Evolve both reference states once, reporting ``n_f`` after every Trotter layer.

    One evolution yields the whole depth ladder, since layer ``d``'s answer is read off the state
    after ``d`` layers -- against ``layer_cliffords[d - 1]``, not the depth-20 Clifford, which is
    the same prefix-nesting that lets the reduction cache be sliced.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--max-bond", type=int, nargs="+", default=[128, 256, 512, 800])
    parser.add_argument("--cutoff", type=float, default=1e-12)
    parser.add_argument(
        "--state",
        choices=("both", "scv", "meson"),
        default="both",
        help="evolve one reference state instead of both. Schrodinger picture needs a separate "
        "evolution per state and they share nothing, so the two halves can run on separate "
        "hosts and be combined from their --out files afterwards.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="write per-depth occupations here as .npy, rewritten after every layer so a run "
        "that is still going can still be read.",
    )
    args = parser.parse_args()

    scv_prep, body = split_state_prep(_CIRCUITS_DIR / "x_100_SCV.qasm")
    meson_prep, _ = split_state_prep(_CIRCUITS_DIR / "x_100_meson.qasm")
    reduced = _reduce_body_cached(
        body, num_qubits=NUM_QUBITS, rz_per_layer=RZ_PER_LAYER, max_layers=args.layers
    )
    layers = [
        fuse_rotations(reduced.rotations[start : start + RZ_PER_LAYER])
        for start in range(0, len(reduced.rotations), RZ_PER_LAYER)
    ]

    for max_bond in args.max_bond:
        header = (
            f"{'depth':>6}  {'n_f':>12}  {'bond':>6}  {'evolve':>9}  {'measure':>9}"
        )
        print(
            f"\nmax_bond={max_bond}  cutoff={args.cutoff:.0e}  "
            f"rotations/layer={len(layers[0]):,}",
            flush=True,
        )
        print(header, flush=True)
        print("-" * len(header), flush=True)
        wanted = [
            (name, prep)
            for name, prep in (("scv", scv_prep), ("meson", meson_prep))
            if args.state in ("both", name)
        ]
        states = {name: _product_state(prep) for name, prep in wanted}
        history: dict[int, dict[str, list[float]]] = {}
        for depth, layer in enumerate(layers, start=1):
            started = perf_counter()
            for psi in states.values():
                _apply_layer(psi, layer, max_bond=max_bond, cutoff=args.cutoff)
            evolved = perf_counter()
            occupations = {
                name: _occupations(psi, reduced.layer_cliffords[depth - 1])
                for name, psi in states.items()
            }
            history[depth] = occupations
            if args.out:
                np.savez(
                    args.out,
                    **{
                        f"{name}_{depth}": np.asarray(values)
                        for depth, entry in history.items()
                        for name, values in entry.items()
                    },
                )
            n_f = (
                sum(
                    site_diff(occupations["meson"], occupations["scv"], r)
                    for r in range(NUM_QUBITS // 2)
                )
                if args.state == "both"
                else float("nan")
            )
            print(
                f"{depth:>6}  {n_f:>12.8f}  "
                f"{max(psi.max_bond() for psi in states.values()):>6}  "
                f"{evolved - started:>8.1f}s  {perf_counter() - evolved:>8.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
