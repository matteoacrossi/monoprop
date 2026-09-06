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

"""Run the SU(2) LSH hadron-dynamics reproduction for a handful of Trotter steps.

Usage (from the repository root -- needs the package-qualified ``-m`` form to resolve the
``benches.hadron`` imports, a plain ``python example.py`` will not):

    ./benches/hadron/fetch_fixtures.sh              # once, to populate .cache/
    uv run --group hadron python -m benches.hadron.example --layers 1 2

Prints n_f(t) for each requested layer count next to the published TN/PP_CPU/PP_GPU/QPU
values from lsh_x100_benchmark.csv, plus wall-clock time and retained term count (T4a's
logging requirement).

``Q(SCV)`` and ``Q(meson)`` are the conserved charge ``60 - sum_r [<Z_i(r)> + <Z_o(r)>]/2``
of each run -- equivalently the total occupation ``sum_j n(j)``. Both are exactly 60 under the
true dynamics, so ``drift`` (the larger deviation of the two) is a correctness invariant: a
nonzero value at shallow depth means something is broken. It is *not* a truncation-error
estimate -- see
[propagate.Run.charge_drift][benches.hadron.propagate.Run.charge_drift].

``--two-qubit-error`` turns on H3's depolarizing damping (see ``noise.py``), which should be
compared against the QPU column rather than the classical ones. ``--two-qubit-error 0`` is its
noiseless control: same layer-by-layer propagation, no channel.

``--basis majorana`` propagates in the Majorana basis instead (see ``majorana.py``), with
``--cutoff-type`` selecting what ``--cutoff`` counts there. Cutoff numbers are *not* comparable
between the bases, so compare error against the ``peak`` column, never at equal cutoff.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benches.hadron.majorana import run as majorana_run
from benches.hadron.propagate import run

_HADRON_DIR = Path(__file__).parent
_CIRCUITS_DIR = (
    _HADRON_DIR
    / ".cache/qat/data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh"
)


def _load_benchmark_row(step: int) -> dict[str, str]:
    with (_HADRON_DIR / "lsh_x100_benchmark.csv").open() as f:
        for row in csv.DictReader(f):
            if int(row["step"]) == step:
                return row
    raise ValueError(f"no benchmark row for step {step}")


def main() -> None:
    """Parse CLI args, propagate n_f(t) for each requested layer count, and print a table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Trotter layer counts to run",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=1000,
        help="weight cutoff: Pauli weight, or --cutoff-type in the majorana basis",
    )
    parser.add_argument(
        "--lower-atol", type=float, default=None, help="coefficient-magnitude cutoff"
    )
    parser.add_argument(
        "--two-qubit-error",
        type=float,
        default=None,
        help="depolarizing error per two-qubit gate (H3); omit for the one-shot noiseless path",
    )
    parser.add_argument(
        "--basis",
        choices=("pauli", "majorana"),
        default="pauli",
        help="propagation basis (section 8); majorana has no noise channel",
    )
    parser.add_argument(
        "--cutoff-type",
        choices=("length", "support"),
        default="length",
        help="what --cutoff counts in the majorana basis; ignored for pauli",
    )
    args = parser.parse_args()

    if args.basis == "majorana" and args.two_qubit_error is not None:
        raise SystemExit(
            "--two-qubit-error is Pauli-side only; H3 has no Majorana path"
        )

    if not (_CIRCUITS_DIR / "x_100_SCV.qasm").exists():
        raise SystemExit(
            f"fixtures missing; run {_HADRON_DIR / 'fetch_fixtures.sh'} first"
        )

    columns = (
        f"{'step':>4}  {'n_f (mine)':>11}  {'TN':>9}  {'PP_GPU':>9}  {'QPU':>9}"
        f"  {'Q(SCV)':>12}  {'Q(meson)':>12}  {'drift':>9}  {'peak':>9}  {'time':>7}"
    )
    print(columns)
    print("-" * len(columns))
    for layers in args.layers:
        if args.basis == "majorana":
            outcome = majorana_run(
                _CIRCUITS_DIR,
                max_layers=layers,
                cutoff=args.cutoff,
                lower_atol=args.lower_atol,
                cutoff_type=args.cutoff_type,
            )
        else:
            outcome = run(
                _CIRCUITS_DIR,
                max_layers=layers,
                cutoff=args.cutoff,
                lower_atol=args.lower_atol,
                two_qubit_error=args.two_qubit_error,
            )
        row = _load_benchmark_row(layers)
        print(
            f"{layers:>4}  {outcome.n_f:>11.6f}  {float(row['TN']):>9.5f}  "
            f"{float(row['PP_GPU']):>9.5f}  {float(row['QPU']):>9.5f}  "
            f"{outcome.charge_scv:>12.8f}  {outcome.charge_meson:>12.8f}  "
            f"{outcome.charge_drift:>9.2e}  {outcome.peak_terms:>9,}  "
            f"{outcome.seconds:>6.1f}s"
        )


if __name__ == "__main__":
    main()
