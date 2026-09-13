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
from qiskit.transpiler.passes import CommutativeOptimization, LitinskiTransformation

import monoprop

NUM_QUBITS = 120
NUM_SITES = NUM_QUBITS // 2

#: Commute every Clifford to the end, then merge rotations that repeat a generator.
REDUCE = PassManager(
    [
        LitinskiTransformation(fix_clifford=True, use_ppr=True, insert_barrier=True),
        CommutativeOptimization(),
    ]
)


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


def reduce(
    body: QuantumCircuit, initial_states: dict[str, tuple[int, ...]]
) -> tuple[dict[str, monoprop.Circuit], Clifford]:
    """Rewrite the body as Pauli rotations, one circuit per reference state.

    Two transpiler passes do the work. [LitinskiTransformation][] commutes all 21,350 Clifford
    gates to the end, turning each ``rz`` into one Pauli product rotation; ``CommutativeOptimization``
    then merges generators that repeat with only commuting rotations between them, which is exact
    -- ``exp(-i a P) exp(-i b P) = exp(-i (a + b) P)`` -- and takes 10,720 rotations down to 8,320.
    Propagation cost tracks the rotation count, so that is worth ~1.28x.

    The rotations are emitted in the DAG's topological order rather than the circuit's. The two
    differ only by transpositions of commuting rotations, so the operator is the same; it does
    mean the order crosses Trotter-layer boundaries, which is harmless only because nothing here
    acts at one. Merging is likewise safe across the whole body for that reason.
    """
    transformed = REDUCE.run(body)
    index = {bit: position for position, bit in enumerate(transformed.qubits)}
    split = next(
        k for k, i in enumerate(transformed.data) if i.operation.name == "barrier"
    )
    rotations, tail = QuantumCircuit(NUM_QUBITS), QuantumCircuit(NUM_QUBITS)
    for k, instruction in enumerate(transformed.data):
        target = rotations if k < split else tail
        if instruction.operation.name != "barrier":
            target.append(
                instruction.operation, [index[bit] for bit in instruction.qubits]
            )
    # The two reference states share every rotation and differ only in what they start from, so
    # the passes and the conversion run once and the gates are handed to both circuits.
    converted = monoprop.from_qiskit_circuit(
        rotations, list(next(iter(initial_states.values())))
    )
    circuits = {
        name: monoprop.Circuit(
            gates=converted.gates,
            system_size=NUM_QUBITS,
            parameters=converted.parameters,
            initial_state=state,
        )
        for name, state in initial_states.items()
    }
    return circuits, Clifford(tail)


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
    circuits, final_clifford = reduce(body, {"SCV": scv_prep, "meson": meson_prep})
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
    print(f"rotations    {len(circuits['SCV'].gates):,}")
    print(f"n_f          {n_f:.6f}")
    print(f"Q(SCV)       {charges['SCV']:.6f}   (exact: {NUM_SITES})")
    print(f"Q(meson)     {charges['meson']:.6f}   (exact: {NUM_SITES})")
    print(f"drift        {max(abs(q - NUM_SITES) for q in charges.values()):.2e}")
    print(f"peak terms   {peak:,}")
    print(f"setup        {prepared - started:.1f}s")
    print(f"propagate    {perf_counter() - prepared:.1f}s")


if __name__ == "__main__":
    main()
