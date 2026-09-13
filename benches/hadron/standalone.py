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

r"""Self-contained reproduction of ``n_f`` and the conserved charge at Trotter depth 20.

One file, two knobs (``--atol`` and ``--cutoff``), no imports from this repository. Everything
qiskit or monoprop already does is delegated to them; what is left here is the problem-specific
part -- the observable and the driver.

**The problem.** SU(2) lattice gauge theory in the loop-string-hadron formulation, 60 sites on
120 qubits at ``x = 100``, from arXiv:2602.18080. The observable is the differential fermion
occupation between two reference states::

    n_f = sum_r (-1)^r [ (n(2r) + n(2r+1))_meson - (n(2r) + n(2r+1))_SCV ]

with ``n(w) = (1 - <Z_w>) / 2`` on output wire ``w``. The two circuits differ only in their
opening layer of ``x`` gates. Total occupation is conserved exactly by the dynamics, so
``Q = sum_w n(w)`` must equal 60 for either state; its deviation is a reference-free error
proxy, reported as ``drift``.

**Why the circuit is rewritten before propagating.** monoprop propagates Pauli rotations, and
the QASM is Clifford+RZ with ``swap``-based routing -- 21,350 Clifford gates against 10,720
``rz``. The Litinski transformation commutes every Clifford to the end of the circuit, turning
each ``rz(theta) q[k]`` into a single Pauli product rotation ``exp(-i theta/2 P)`` and leaving
one final Clifford ``D`` behind. The observable is then reduced through that same ``D``:
``<Z_w>`` at the output is ``<D^dag Z_w D>`` evaluated on the rotations alone.

This is not a workaround for a missing feature. A Clifford maps each Pauli term to one Pauli
term, so applying one to the operator is cheap per term but costs O(terms) -- and the operator
here holds 10^5 to 10^7 terms, against a 120-qubit tableau. Folding the Cliffords into a frame
first is the asymptotically right algorithm, and it is exact, which leaves ``--atol`` as the
only approximation in the calculation. qiskit's pass is Rust-backed and does the whole body in
0.04s.

**Cost.** Setup is ~1s, independent of ``--atol``. Propagation is 120 wires x 2 reference
states, each independent, and is where the budget goes: ~87s at ``--atol 1e-6`` on 36 cores.
The value converges as ``--atol`` tightens; 1e-8 gives ``n_f = 0.116864``, ``drift = 3.6e-06``.

Usage::

    python standalone.py --atol 1e-6
    python standalone.py --atol 1e-8 --circuits /path/to/qasm/dir
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Clifford, Pauli, SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import LitinskiTransformation

import monoprop

NUM_QUBITS = 120
NUM_SITES = NUM_QUBITS // 2

Rotation = tuple[monoprop.Pauli, float]


def load(path: Path) -> tuple[tuple[int, ...], QuantumCircuit]:
    """Parse a circuit into its state-prep qubits and the Trotter body.

    ``swap`` keeps qiskit's strict QASM 2 parser from recognising the file, hence the legacy
    instruction set. A qubit is occupied when an *odd* number of leading ``x`` gates touch it,
    not when it merely appears: the meson file applies ``x`` to q[59] twice, so counting presence
    rather than parity would wrongly call q[59] occupied in both files instead of only in SCV.
    """
    circuit = qasm2.load(
        str(path), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
    )
    index = {bit: position for position, bit in enumerate(circuit.qubits)}
    end = 0
    while circuit.data[end].operation.name == "x":
        end += 1
    flips: dict[int, int] = {}
    for instruction in circuit.data[:end]:
        qubit = index[instruction.qubits[0]]
        flips[qubit] = flips.get(qubit, 0) + 1
    body = QuantumCircuit(NUM_QUBITS)
    for instruction in circuit.data[end:]:
        body.append(instruction.operation, [index[bit] for bit in instruction.qubits])
    return tuple(sorted(q for q, n in flips.items() if n % 2 == 1)), body


def to_rotations(body: QuantumCircuit) -> tuple[list[Rotation], Clifford]:
    """Commute every Clifford to the end, returning the rotations and that final Clifford.

    Each rotation comes back as ``(generator, signed angle)`` on the full register. The pass
    emits them in the DAG's topological order rather than the QASM's sequential one; the two
    orders differ only by transpositions of commuting rotations, so the product is the same.
    """
    transformed = PassManager(
        [LitinskiTransformation(fix_clifford=True, use_ppr=True, insert_barrier=True)]
    ).run(body)
    index = {bit: position for position, bit in enumerate(transformed.qubits)}
    split = next(
        k for k, i in enumerate(transformed.data) if i.operation.name == "barrier"
    )

    rotations = []
    for instruction in transformed.data[:split]:
        label = instruction.operation.pauli().to_label()
        sign = -1.0 if label.startswith("-") else 1.0
        label = label.lstrip("+-")
        qubits = [index[bit] for bit in instruction.qubits]
        # a qiskit label runs last-operand-first over the gate's own qubits
        letters = [
            (qubits[len(label) - 1 - k], letter)
            for k, letter in enumerate(label)
            if letter != "I"
        ]
        pauli = monoprop.Pauli(
            "".join(letter for _, letter in letters),
            tuple(qubit for qubit, _ in letters),
        )
        rotations.append((pauli, sign * float(instruction.operation.params[0])))

    tail = QuantumCircuit(NUM_QUBITS)
    for instruction in transformed.data[split + 1 :]:
        tail.append(instruction.operation, [index[bit] for bit in instruction.qubits])
    return rotations, Clifford(tail)


def fuse(rotations: list[Rotation]) -> list[Rotation]:
    """Merge repeated generators separated only by rotations that commute with them.

    Exact, not an approximation: ``exp(-i a P) exp(-i b P) = exp(-i (a + b) P)``, and a rotation
    slides past any rotation whose generator commutes with it -- two Paulis commute when they
    differ on an even number of shared qubits. Worth doing because propagation cost tracks the
    rotation count. Fusing across the whole body is only safe because nothing here acts at a
    Trotter boundary; with a noise channel between layers it would have to be done per layer.
    """
    fused: list[Rotation] = []
    keys: list[dict[int, str]] = []
    for pauli, angle in rotations:
        key = dict(zip(pauli.qubits, pauli.string, strict=True))
        for i in reversed(range(len(fused))):
            if keys[i] == key:
                total = fused[i][1] + angle
                if total == 0.0:
                    del fused[i], keys[i]
                else:
                    fused[i] = (fused[i][0], total)
                break
            if sum(1 for q, p in key.items() if keys[i].get(q, p) != p) % 2:
                fused.append((pauli, angle))
                keys.append(key)
                break
        else:
            fused.append((pauli, angle))
            keys.append(key)
    return fused


def build_circuit(
    rotations: list[Rotation], initial_state: tuple[int, ...]
) -> monoprop.Circuit:
    """A monoprop circuit of ExpGates, one per rotation.

    ExpGate applies ``exp(+i theta H)``, so ``H = -0.5 P`` driven by the signed angle reproduces
    the ``exp(-i theta P / 2)`` the Pauli product rotations denote.
    """
    return monoprop.Circuit(
        gates=tuple(
            monoprop.ExpGate(monoprop.PauliOperator({p: -0.5}, num_qubits=NUM_QUBITS))
            for p, _ in rotations
        ),
        system_size=NUM_QUBITS,
        parameters=tuple(angle for _, angle in rotations),
        initial_state=initial_state,
    )


def observable(wire: int, inverse_clifford: Clifford) -> monoprop.PauliOperator:
    """``D^dag Z_wire D`` as a one-term operator, given ``D^-1``.

    Asking for it as ``evolve(D, frame="h")`` makes qiskit rebuild a 120-qubit adjoint on every
    call; ``evolve(D^-1, frame="s")`` is the same operator read straight off the tableau.
    monoprop's converter handles the qubit-ordering flip between the two libraries.
    """
    label = ["I"] * NUM_QUBITS
    label[NUM_QUBITS - 1 - wire] = "Z"
    evolved = Pauli("".join(label)).evolve(inverse_clifford, frame="s")
    return monoprop.from_qiskit_operator(SparsePauliOp(evolved))


def main() -> None:
    """Reduce both circuits, propagate every wire, and report n_f, the charge and the cost."""
    parser = argparse.ArgumentParser(
        description="n_f at Trotter depth 20, SU(2) LSH, x=100"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-6, help="coefficient truncation"
    )
    parser.add_argument(
        "--cutoff", type=int, default=1000, help="Pauli-weight truncation"
    )
    parser.add_argument("--circuits", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    started = perf_counter()
    scv_prep, body = load(args.circuits / "x_100_SCV.qasm")
    meson_prep, _ = load(args.circuits / "x_100_meson.qasm")
    rotations, final_clifford = to_rotations(body)
    rotations = fuse(rotations)
    circuits = {
        "SCV": build_circuit(rotations, scv_prep),
        "meson": build_circuit(rotations, meson_prep),
    }
    inverse_clifford = final_clifford.adjoint()
    prepared = perf_counter()

    occupation = {name: [0.0] * NUM_QUBITS for name in circuits}
    peak = 0
    for wire in range(NUM_QUBITS):
        measured = observable(wire, inverse_clifford)
        for name, circuit in circuits.items():
            propagator = monoprop.PauliPropagator.from_circuit(
                circuit, measured, cutoff=args.cutoff, lower_atol=args.atol
            )
            occupation[name][wire] = (1 - propagator.expval()) / 2
            peak = max(peak, propagator.size())

    n_f = sum(
        (-1) ** r
        * (
            occupation["meson"][2 * r]
            + occupation["meson"][2 * r + 1]
            - occupation["SCV"][2 * r]
            - occupation["SCV"][2 * r + 1]
        )
        for r in range(NUM_SITES)
    )
    charges = {name: sum(values) for name, values in occupation.items()}

    print(f"atol         {args.atol:.0e}")
    print(f"cutoff       {args.cutoff}")
    print(f"rotations    {len(rotations):,}")
    print(f"n_f          {n_f:.6f}")
    print(f"Q(SCV)       {charges['SCV']:.6f}   (exact: {NUM_SITES})")
    print(f"Q(meson)     {charges['meson']:.6f}   (exact: {NUM_SITES})")
    print(f"drift        {max(abs(q - NUM_SITES) for q in charges.values()):.2e}")
    print(f"peak terms   {peak:,}")
    print(f"setup        {prepared - started:.1f}s")
    print(f"propagate    {perf_counter() - prepared:.1f}s")


if __name__ == "__main__":
    main()
