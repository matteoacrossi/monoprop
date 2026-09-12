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

r"""Wire-parallel ``lower_atol`` sweep across MPI ranks (T5's tolerance-vs-cost study).

The 120 output wires are independent -- each is its own backpropagation -- so they distribute
with no communication at all until one reduction of the occupation arrays at the end. That is a
different axis from monoprop's own MPI, which splits a single operator across ranks: here every
rank evolves whole wires by itself and ranks are only a way to hand them out. Measured, splitting
one wire would not pay anyway -- a wire saturates at about 16 partitions, and this circuit's cost
is set by its 10,720 rotations rather than by the terms they carry.

Every propagator is therefore built with ``comm=MPI.COMM_SELF``, and that is load-bearing rather
than decorative. Leaving ``comm`` unset under ``mpirun`` does *not* give a private propagator: the
ranks land in one communicator and distribute the operator between them, so each rank keeps about
half the terms of the wire it thinks it owns while the collectives mix wires that have nothing to
do with each other. It fails silently, into plausible-looking numbers -- at 20 layers and
``atol=1e-2`` it moved ``n_f`` from 0.118 to 0.833 and halved the retained term count, with no
error raised. ``COMM_SELF`` reproduces the single-process result bit for bit.

Wires are handed out strided rather than in blocks. Retained terms follow a symmetric tent across
the lattice -- about 9k at the edge wires against 60k in the middle at ``atol=1e-5`` -- so a
strided split gives every rank the same mix of cheap and expensive wires.

Bind ranks explicitly. Open MPI binds to a single core by default at small ``-np``, which leaves
monoprop one partition instead of eighteen and costs about 3x; ``--bind-to none`` for one rank per
node, or ``--map-by ppr:N:node:PE=M --bind-to core`` to give co-located ranks disjoint cores.

Usage (from the repository root, with an MPI-enabled build at the same path on every host)::

    mpirun -H host1:18,host2:18 -np 2 --map-by ppr:1:node --bind-to none \\
        --wdir /home/ubuntu/monoprop \\
        /home/ubuntu/monoprop/.venv/bin/python -m benches.hadron.mpi_sweep --atols 1e-5 1e-6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

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

_HADRON_DIR = Path(__file__).parent
_CIRCUITS_DIR = (
    _HADRON_DIR
    / ".cache/qat/data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh"
)


def main() -> None:
    """Sweep ``lower_atol`` with the wires split across ranks, printing one row each on rank 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--cutoff", type=int, default=1000)
    parser.add_argument("--atols", type=float, nargs="+", required=True)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    scv_prep, body = split_state_prep(_CIRCUITS_DIR / "x_100_SCV.qasm")
    meson_prep, _ = split_state_prep(_CIRCUITS_DIR / "x_100_meson.qasm")
    reduced = _reduce_body_cached(
        body,
        num_qubits=NUM_QUBITS,
        rz_per_layer=RZ_PER_LAYER,
        max_layers=args.layers,
    )
    rotations = [
        rotation
        for start in range(0, len(reduced.rotations), RZ_PER_LAYER)
        for rotation in fuse_rotations(reduced.rotations[start : start + RZ_PER_LAYER])
    ]
    circuits = {
        state: build_circuit(rotations, num_qubits=NUM_QUBITS, initial_state=prep)
        for state, prep in (("scv", scv_prep), ("meson", meson_prep))
    }
    wires = range(rank, NUM_QUBITS, size)

    if rank == 0:
        header = (
            f"{'lower_atol':>11}  {'n_f':>11}  {'drift':>9}  {'peak':>11}  "
            f"{'total':>13}  {'time':>8}"
        )
        print(f"ranks={size}  layers={args.layers}  cutoff={args.cutoff}", flush=True)
        print(header, flush=True)
        print("-" * len(header), flush=True)

    for atol in sorted(args.atols, reverse=True):
        comm.Barrier()
        started = MPI.Wtime()
        occupations = {state: np.zeros(NUM_QUBITS) for state in circuits}
        peak = 0
        total = 0
        for wire in wires:
            pauli, sign = reduce_output_pauli(reduced.final_clifford, wire, NUM_QUBITS)
            operator = monoprop.PauliOperator({pauli: sign}, num_qubits=NUM_QUBITS)
            for state, circuit in circuits.items():
                propagator = monoprop.PauliPropagator.from_circuit(
                    circuit,
                    operator,
                    cutoff=args.cutoff,
                    lower_atol=atol,
                    comm=MPI.COMM_SELF,
                )
                occupations[state][wire] = (1 - propagator.expval()) / 2
                if state == "scv":
                    # Heisenberg evolution ignores the reference state, so both runs of this wire
                    # retain the same terms; counting one keeps the totals comparable to a
                    # single-process run.
                    size_here = propagator.size()
                    peak = max(peak, size_here)
                    total += size_here

        gathered = {
            state: np.zeros(NUM_QUBITS) if rank == 0 else None for state in occupations
        }
        for state, values in occupations.items():
            comm.Reduce(values, gathered[state], op=MPI.SUM, root=0)
        peak = comm.reduce(peak, op=MPI.MAX, root=0)
        total = comm.reduce(total, op=MPI.SUM, root=0)
        elapsed = MPI.Wtime() - started

        if rank == 0:
            per_site = [
                site_diff(gathered["meson"], gathered["scv"], r)
                for r in range(NUM_QUBITS // 2)
            ]
            drift = max(
                abs(gathered[state].sum() - NUM_QUBITS / 2) for state in gathered
            )
            print(
                f"{atol:>11.1e}  {sum(per_site):>11.6f}  {drift:>9.2e}  "
                f"{peak:>11,}  {total:>13,}  {elapsed:>7.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
