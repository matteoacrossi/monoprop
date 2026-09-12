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

r"""Measure the bias of magnitude truncation against an unbiased estimator of the same budget.

Discarding every term below ``atol`` is not a neutral act: it only ever removes weight, so the
question is whether what it removes cancels. This answers it without any external reference, by
comparing three boundary rules that share one engine budget, so only the rule differs.

- ``none`` -- no boundary truncation at all. With the engine budget tight, this is the answer the
  other two are trying to reproduce.
- ``drop`` -- discard ``|c| < atol``, which is what [noise.dropped][benches.hadron.noise.dropped]
  and the engine's own ``lower_atol`` do.
- ``roulette`` -- keep a sub-threshold term with probability ``|c| / atol`` and rescale it to
  ``sign(c) * atol``. Its expectation is ``(|c| / atol) * atol * sign(c) = c``, so the estimator
  is unbiased term by term, and averaging realizations converges on ``none`` no matter how coarse
  ``atol`` is. Variance is the price.

So ``drop - none`` is the bias, and ``roulette - none`` must vanish within the standard error. If
instead roulette also lands on ``drop``, the difference is not truncation bias and the deviation
lives somewhere else.

Realizations are distributed one per rank. The reference legs are computed on rank 0 while the
others sample, so a run costs about what its slowest leg costs.

Usage::

    mpirun -H host1:18,host2:18 -np 36 --map-by ppr:18:node:PE=1 --bind-to core \
        --wdir /home/ubuntu/monoprop \
        /home/ubuntu/monoprop/.venv/bin/python -m benches.hadron.roulette \
        --wire 60 --atol 1e-3 --engine-atol 1e-6 --realizations 72
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

import monoprop
from benches.hadron import noise
from benches.hadron.observable import basis_expectation
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


def rouletted(
    operator: monoprop.PauliOperator, atol: float, rng: np.random.Generator
) -> monoprop.PauliOperator:
    """Russian-roulette counterpart of [noise.dropped][benches.hadron.noise.dropped].

    A term at or above ``atol`` is kept as it is. Below it, the term survives with probability
    ``|c| / atol`` and is rescaled to ``sign(c) * atol`` when it does, which leaves its
    expectation equal to ``c``. Nothing is removed on average, so repeated realizations estimate
    the untruncated operator rather than a systematically lighter one.
    """
    terms = {}
    for term, coefficient in operator.terms.items():
        magnitude = abs(coefficient)
        if magnitude >= atol:
            terms[term] = coefficient
        elif rng.random() < magnitude / atol:
            terms[term] = atol if coefficient > 0 else -atol
    return monoprop.PauliOperator(terms, num_qubits=operator.num_qubits)


def _propagate(
    layer_circuits: list[monoprop.Circuit],
    initial_operator: monoprop.PauliOperator,
    occupied: set[int],
    *,
    engine_atol: float,
    atol: float | None,
    rng: np.random.Generator | None,
) -> tuple[float, int]:
    """Backpropagate one wire, truncating at each layer boundary by the chosen rule.

    ``atol`` of ``None`` applies no boundary rule; ``rng`` of ``None`` selects the deterministic
    drop over roulette. Returns ``(expectation, peak terms)``.
    """
    operator = initial_operator
    peak = 0
    for index in reversed(range(len(layer_circuits))):
        if atol is not None:
            operator = (
                noise.dropped(operator, atol)
                if rng is None
                else rouletted(operator, atol, rng)
            )
        propagator = monoprop.PauliPropagator.from_circuit(
            layer_circuits[index],
            operator,
            cutoff=1000,
            lower_atol=engine_atol,
            comm=MPI.COMM_SELF,
        )
        peak = max(peak, propagator.size())
        operator = propagator.evolved_operator()
    return basis_expectation(operator, occupied), peak


def main() -> None:
    """Compare the deterministic drop against the unbiased estimator at the same budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--wire", type=int, default=60)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--engine-atol", type=float, default=1e-6)
    parser.add_argument("--realizations", type=int, default=72)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    scv_prep, body = split_state_prep(_CIRCUITS_DIR / "x_100_SCV.qasm")
    reduced = _reduce_body_cached(
        body, num_qubits=NUM_QUBITS, rz_per_layer=RZ_PER_LAYER, max_layers=args.layers
    )
    layer_circuits = [
        build_circuit(
            fuse_rotations(reduced.rotations[start : start + RZ_PER_LAYER]),
            num_qubits=NUM_QUBITS,
            initial_state=scv_prep,
        )
        for start in range(0, len(reduced.rotations), RZ_PER_LAYER)
    ]
    pauli, sign = reduce_output_pauli(reduced.final_clifford, args.wire, NUM_QUBITS)
    initial_operator = monoprop.PauliOperator({pauli: sign}, num_qubits=NUM_QUBITS)
    occupied = set(scv_prep)

    def run(atol: float | None, rng: np.random.Generator | None) -> tuple[float, int]:
        return _propagate(
            layer_circuits,
            initial_operator,
            occupied,
            engine_atol=args.engine_atol,
            atol=atol,
            rng=rng,
        )

    reference = dropped = None
    if rank == 0:
        started = MPI.Wtime()
        reference = run(None, None)
        dropped = run(args.atol, None)
        print(
            f"wire={args.wire} layers={args.layers} atol={args.atol:.0e} "
            f"engine_atol={args.engine_atol:.0e} realizations={args.realizations} "
            f"ranks={size}",
            flush=True,
        )
        print(
            f"  none      : {reference[0]:+.9f}   peak={reference[1]:>9,}  "
            f"({MPI.Wtime() - started:.0f}s for both reference legs)",
            flush=True,
        )
        print(f"  drop      : {dropped[0]:+.9f}   peak={dropped[1]:>9,}", flush=True)

    samples = []
    peaks = []
    for index in range(rank, args.realizations, size):
        value, peak = run(args.atol, np.random.default_rng(args.seed + index))
        samples.append(value)
        peaks.append(peak)
    gathered = comm.gather(samples, root=0)
    gathered_peaks = comm.gather(peaks, root=0)

    if rank == 0:
        values = np.array([v for chunk in gathered for v in chunk])
        all_peaks = np.array([p for chunk in gathered_peaks for p in chunk])
        mean = values.mean()
        stderr = values.std(ddof=1) / np.sqrt(values.size)
        bias = dropped[0] - reference[0]
        residual = mean - reference[0]
        print(
            f"  roulette  : {mean:+.9f} +- {stderr:.9f}  (n={values.size}, "
            f"mean peak={all_peaks.mean():,.0f})",
            flush=True,
        )
        print(f"\n  drop - none      = {bias:+.3e}", flush=True)
        print(
            f"  roulette - none  = {residual:+.3e}  ({abs(residual) / stderr:.1f} sigma)",
            flush=True,
        )
        verdict = (
            "drop is biased; roulette recovers the untruncated value"
            if abs(bias) > 3 * stderr and abs(residual) < 3 * stderr
            else "inconclusive at this sample size"
        )
        print(f"  verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
