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

r"""Smallest single-process run of ``n_f`` at depth 20, spelled out end to end.

[propagate.run][benches.hadron.propagate.run] already does all of this, so the two-line version
is::

    from benches.hadron.propagate import run

    outcome = run(circuits_dir, max_layers=20, lower_atol=1e-6)

What follows instead unrolls that call, because the interesting part of driving
[PauliPropagator][monoprop.pauli_propagator.PauliPropagator] on this problem is the four steps
between the qasm file and the propagator, not the propagation:

1. **Split the state preparation off the body.** The two reference states differ only in their
   opening layer of ``x`` gates; everything after it is the same circuit.
2. **Reduce the body to Pauli rotations.** Clifford conjugation pushes every Clifford gate to the
   end, leaving ``D_final * R_M ... R_1`` where each ``R_i`` has weight at most 2. This is by far
   the slowest step at ~200s, so it is cached to disk and costs ~0.05s on any later run.
3. **Fuse repeated rotations.** ``exp(-iaP) exp(-ibP) = exp(-i(a+b)P)`` when nothing between them
   fails to commute, which is exact and takes 536 rotations per layer down to 416.
4. **Backpropagate one wire at a time.** ``<Z_w>`` is measured by evolving ``D_final^dag Z_w
   D_final`` backwards through the body, once per reference state.

There is no MPI anywhere here. The engine still uses every core through its own threading -- the
wires are simply walked in sequence rather than dealt out to ranks. Distributing step 4 across
ranks is what [mpi_sweep][benches.hadron.mpi_sweep] does, and doing so requires passing
``comm=MPI.COMM_SELF`` to keep each rank's propagator private; omitting it silently splits one
wire's operator across ranks and returns a wrong answer.

Usage::

    uv run --no-sync python -m benches.hadron.mwe
    uv run --no-sync python -m benches.hadron.mwe --layers 3 --lower-atol 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import monoprop
from benches.hadron.observable import site_diff
from benches.hadron.pauli_frontend import (
    build_circuit,
    fuse_rotations,
    reduce_output_pauli,
)
from benches.hadron.propagate import (
    NUM_QUBITS,
    RZ_PER_LAYER,
    _reduce_body_cached,
)
from benches.hadron.qasm_frontend import split_state_prep

_CIRCUITS_DIR = (
    Path(__file__).parent
    / ".cache/qat/data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh"
)


def main() -> None:
    """Report ``n_f``, the conserved charge and the term counts for one truncation setting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--lower-atol", type=float, default=1e-6)
    parser.add_argument("--cutoff", type=int, default=1000)
    parser.add_argument("--circuits-dir", type=Path, default=_CIRCUITS_DIR)
    args = parser.parse_args()

    started = perf_counter()

    # 1. the two reference states share a body and differ only in their opening x gates
    scv_prep, body = split_state_prep(args.circuits_dir / "x_100_SCV.qasm")
    meson_prep, _ = split_state_prep(args.circuits_dir / "x_100_meson.qasm")

    # 2. and 3. Clifford reduction (cached) and exact per-layer fusion
    reduced = _reduce_body_cached(
        body, num_qubits=NUM_QUBITS, rz_per_layer=RZ_PER_LAYER, max_layers=args.layers
    )
    rotations = [
        rotation
        for start in range(0, len(reduced.rotations), RZ_PER_LAYER)
        for rotation in fuse_rotations(reduced.rotations[start : start + RZ_PER_LAYER])
    ]
    circuits = {
        "scv": build_circuit(rotations, num_qubits=NUM_QUBITS, initial_state=scv_prep),
        "meson": build_circuit(
            rotations, num_qubits=NUM_QUBITS, initial_state=meson_prep
        ),
    }
    prepared = perf_counter()

    # 4. one backpropagation per wire per reference state
    occupations = {name: [0.0] * NUM_QUBITS for name in circuits}
    peak = 0
    total = 0
    for wire in range(NUM_QUBITS):
        pauli, sign = reduce_output_pauli(reduced.final_clifford, wire, NUM_QUBITS)
        observable = monoprop.PauliOperator({pauli: sign}, num_qubits=NUM_QUBITS)
        for index, (name, circuit) in enumerate(circuits.items()):
            propagator = monoprop.PauliPropagator.from_circuit(
                circuit, observable, cutoff=args.cutoff, lower_atol=args.lower_atol
            )
            # <Z_w> -> occupation n(w); expval() contracts in the engine, which is an order of
            # magnitude cheaper than pulling the retained terms back into Python.
            occupations[name][wire] = (1 - propagator.expval()) / 2
            # Heisenberg evolution never reads the reference state, so both propagators retain
            # exactly the same terms -- counting one of them keeps this comparable to the sweep.
            if index == 0:
                peak = max(peak, propagator.size())
                total += propagator.size()

    per_site = [
        site_diff(occupations["meson"], occupations["scv"], r)
        for r in range(NUM_QUBITS // 2)
    ]
    # the dynamics conserves total occupation exactly, so this is a reference-free error proxy
    drift = max(abs(sum(values) - NUM_QUBITS / 2) for values in occupations.values())

    print(f"layers      {args.layers}")
    print(f"lower_atol  {args.lower_atol:.0e}")
    print(f"n_f         {sum(per_site):.6f}")
    print(f"drift       {drift:.2e}")
    print(f"peak terms  {peak:,}")
    print(f"total terms {total:,}")
    print(f"setup       {prepared - started:.1f}s")
    print(f"propagate   {perf_counter() - prepared:.1f}s")


if __name__ == "__main__":
    main()
