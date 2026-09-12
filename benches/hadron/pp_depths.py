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

"""Converged Pauli-propagation ``n_f`` at every depth, as the reference for the MPS ladder.

Slices the cached 20-layer reduction rather than re-reducing per depth: ``rotations`` and
``layer_cliffords`` are prefix-nested, so depth ``d`` is the first ``d * RZ_PER_LAYER`` rotations
read against ``layer_cliffords[d - 1]``.
"""

from __future__ import annotations

import argparse

import numpy as np
from mpi4py import MPI

import monoprop
from benches.hadron.mpi_sweep import _CIRCUITS_DIR
from benches.hadron.observable import site_diff
from benches.hadron.pauli_frontend import (
    build_circuit,
    fuse_rotations,
    reduce_output_pauli,
)
from benches.hadron.propagate import NUM_QUBITS, RZ_PER_LAYER, _reduce_body_cached
from benches.hadron.qasm_frontend import split_state_prep


def main() -> None:
    """Print converged ``n_f`` per depth, with the atol ladder that establishes convergence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--atols", type=float, nargs="+", default=[1e-7, 1e-8])
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    scv_prep, body = split_state_prep(_CIRCUITS_DIR / "x_100_SCV.qasm")
    meson_prep, _ = split_state_prep(_CIRCUITS_DIR / "x_100_meson.qasm")
    reduced = _reduce_body_cached(
        body, num_qubits=NUM_QUBITS, rz_per_layer=RZ_PER_LAYER, max_layers=20
    )

    if rank == 0:
        print(
            f"{'depth':>6}  " + "  ".join(f"{a:>13.0e}" for a in args.atols), flush=True
        )
        print("-" * (6 + 15 * len(args.atols)), flush=True)

    for depth in args.depths:
        rotations = [
            rotation
            for start in range(0, depth * RZ_PER_LAYER, RZ_PER_LAYER)
            for rotation in fuse_rotations(
                reduced.rotations[start : start + RZ_PER_LAYER]
            )
        ]
        clifford = reduced.layer_cliffords[depth - 1]
        circuits = {
            name: build_circuit(rotations, num_qubits=NUM_QUBITS, initial_state=prep)
            for name, prep in (("scv", scv_prep), ("meson", meson_prep))
        }
        row = []
        for atol in args.atols:
            occupations = {name: np.zeros(NUM_QUBITS) for name in circuits}
            for wire in range(rank, NUM_QUBITS, size):
                pauli, sign = reduce_output_pauli(clifford, wire, NUM_QUBITS)
                operator = monoprop.PauliOperator({pauli: sign}, num_qubits=NUM_QUBITS)
                for name, circuit in circuits.items():
                    propagator = monoprop.PauliPropagator.from_circuit(
                        circuit,
                        operator,
                        cutoff=1000,
                        lower_atol=atol,
                        comm=MPI.COMM_SELF,
                    )
                    occupations[name][wire] = (1 - propagator.expval()) / 2
            gathered = {
                name: (np.zeros(NUM_QUBITS) if rank == 0 else None)
                for name in occupations
            }
            for name, values in occupations.items():
                comm.Reduce(values, gathered[name], op=MPI.SUM, root=0)
            if rank == 0:
                row.append(
                    sum(
                        site_diff(gathered["meson"], gathered["scv"], r)
                        for r in range(NUM_QUBITS // 2)
                    )
                )
        if rank == 0:
            print(f"{depth:>6}  " + "  ".join(f"{v:>13.8f}" for v in row), flush=True)


if __name__ == "__main__":
    main()
