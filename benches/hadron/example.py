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
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from benches.hadron.propagate import n_f_at_layer

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
    parser.add_argument("--cutoff", type=int, default=1000, help="Pauli-weight cutoff")
    parser.add_argument(
        "--lower-atol", type=float, default=None, help="coefficient-magnitude cutoff"
    )
    args = parser.parse_args()

    if not (_CIRCUITS_DIR / "x_100_SCV.qasm").exists():
        raise SystemExit(
            f"fixtures missing; run {_HADRON_DIR / 'fetch_fixtures.sh'} first"
        )

    header = f"{'step':>4}  {'n_f (mine)':>12}  {'TN':>10}  {'PP_CPU':>10}  {'PP_GPU':>10}  {'QPU':>10}  {'time':>7}"
    print(header)
    print("-" * len(header))
    for layers in args.layers:
        t0 = time.time()
        n_f, _ = n_f_at_layer(
            _CIRCUITS_DIR,
            max_layers=layers,
            cutoff=args.cutoff,
            lower_atol=args.lower_atol,
        )
        elapsed = time.time() - t0
        row = _load_benchmark_row(layers)
        print(
            f"{layers:>4}  {n_f:>12.6f}  {float(row['TN']):>10.6f}  "
            f"{float(row['PP_CPU']):>10.6f}  {float(row['PP_GPU']):>10.6f}  "
            f"{float(row['QPU']):>10.6f}  {elapsed:>6.1f}s"
        )


if __name__ == "__main__":
    main()
