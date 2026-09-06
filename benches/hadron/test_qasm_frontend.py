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

"""T2 acceptance criteria from PLAN_lsh_pauli_prop.md section 6."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from benches.hadron.qasm_frontend import _gate_lines, gate_histogram, parse_lsh_qasm

EXPECTED_GATE_HISTOGRAM_COMMON = {
    "cx": 11840,
    "rz": 10720,
    "h": 4720,
    "swap": 2330,
}


def test_gate_histogram_scv(circuits_dir: Path) -> None:
    histogram = gate_histogram(circuits_dir / "x_100_SCV.qasm")
    assert histogram == {**EXPECTED_GATE_HISTOGRAM_COMMON, "x": 2460}


def test_gate_histogram_meson(circuits_dir: Path) -> None:
    histogram = gate_histogram(circuits_dir / "x_100_meson.qasm")
    assert histogram == {**EXPECTED_GATE_HISTOGRAM_COMMON, "x": 2462}


def test_multiset_diff_is_only_the_two_meson_x_gates(circuits_dir: Path) -> None:
    scv_lines = sorted(_gate_lines(circuits_dir / "x_100_SCV.qasm"))
    meson_lines = sorted(_gate_lines(circuits_dir / "x_100_meson.qasm"))

    # Every SCV line has a matching meson line; the leftover meson lines are the extra x gates.
    scv_counts = Counter(scv_lines)
    meson_counts = Counter(meson_lines)
    extra_in_meson = meson_counts - scv_counts
    extra_in_scv = scv_counts - meson_counts

    assert extra_in_scv == {}
    assert extra_in_meson == {"x q[59];": 1, "x q[60];": 1}


def test_all_twenty_layers_identical_after_permutation_tracking(
    circuits_dir: Path,
) -> None:
    parsed = parse_lsh_qasm(circuits_dir / "x_100_SCV.qasm")
    assert len(parsed.layers) == 20
    first_layer = parsed.layers[0]
    assert all(layer == first_layer for layer in parsed.layers[1:])
    assert len(first_layer) == 1484
