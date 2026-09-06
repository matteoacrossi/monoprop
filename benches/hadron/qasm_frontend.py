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

"""QASM front end for the SU(2) LSH hadron-dynamics circuits (T2 of PLAN_lsh_pauli_prop.md).

Parses the OPENQASM 2.0 circuits, splits off the state-prep ``x`` gates, and segments the
Trotter body into layers -- tracking the physical -> logical permutation induced by ``swap``
so that gates are reported on the logical qubits they actually act on. This module only
translates gate *labels*; it does not reduce Clifford gates into Pauli-rotation generators --
see ``pauli_frontend.py`` for that (T1's front-end decision).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_QUBIT_RE = re.compile(r"q\[(\d+)\]")
_RZ_ANGLE_RE = re.compile(r"rz\(([^)]+)\)")
_HEADER_PREFIXES = ("OPENQASM", "include", "qreg")


@dataclass(frozen=True, slots=True)
class LogicalGate:
    """One gate translated onto the logical qubits it acts on.

    Attributes:
        name: Gate name -- ``"x"``, ``"h"``, ``"cx"``, or ``"rz"``. ``swap`` gates never appear
            here; they update the permutation instead of being emitted.
        qubits: Logical qubit indices, in the QASM operand order (``[control, target]`` for
            ``cx``).
        angle: The rotation angle for ``rz`` gates; ``None`` otherwise.
    """

    name: str
    qubits: tuple[int, ...]
    angle: float | None = None


@dataclass(frozen=True, slots=True)
class ParsedLshCircuit:
    """A parsed LSH hadron-dynamics circuit: state prep plus Trotter layers.

    Attributes:
        num_qubits: Circuit width.
        state_prep: Physical qubit indices set to |1> by the leading ``x`` gates. Physical and
            logical coincide here, since no ``swap`` has run yet.
        layers: One tuple of [LogicalGate][] per Trotter layer, in circuit order, translated
            through the physical -> logical permutation active at each gate.
        final_permutation: ``final_permutation[physical]`` is the logical qubit sitting at that
            physical position after all included layers.
    """

    num_qubits: int
    state_prep: tuple[int, ...]
    layers: tuple[tuple[LogicalGate, ...], ...]
    final_permutation: tuple[int, ...]


def _gate_lines(path: str | Path) -> list[str]:
    """Return the non-blank, non-header lines of a QASM 2.0 file."""
    lines = [
        line.strip() for line in Path(path).read_text().splitlines() if line.strip()
    ]
    return [line for line in lines if not line.startswith(_HEADER_PREFIXES)]


def gate_name(line: str) -> str:
    """Return a QASM gate line's name, e.g. ``"rz"`` from ``"rz(-0.15) q[1];"``."""
    return line.split(maxsplit=1)[0].split("(", maxsplit=1)[0]


def gate_qubits(line: str) -> tuple[int, ...]:
    """Return a QASM gate line's qubit operands, in QASM order."""
    return tuple(int(q) for q in _QUBIT_RE.findall(line))


def gate_histogram(path: str | Path) -> dict[str, int]:
    """Count gates by name in a QASM file, for T2's acceptance criterion 1.

    Args:
        path: Path to the QASM file.

    Returns:
        A mapping from gate name to occurrence count.
    """
    counts: dict[str, int] = {}
    for line in _gate_lines(path):
        name = gate_name(line)
        counts[name] = counts.get(name, 0) + 1
    return counts


def split_state_prep(path: str | Path) -> tuple[tuple[int, ...], list[str]]:
    """Split a QASM file's gate lines into the leading state-prep ``x`` run and the body.

    The meson file's leading run applies ``x`` to q[59] *twice* (once in the run shared with
    the SCV file, once more at the very end) alongside the expected new ``x q[60]`` -- so a
    qubit is occupied by an *odd* count of leading ``x`` gates, not by mere presence in the
    run; a naive "which qubits does x touch" reading would wrongly call q[59] occupied in both
    files instead of only in SCV.

    Args:
        path: Path to the QASM file.

    Returns:
        The occupied physical qubit indices after the leading ``x`` run (sorted, deduplicated
        by XOR parity -- physical and logical coincide here, since no ``swap`` has run yet),
        and the remaining raw gate lines (including ``swap``).
    """
    gate_lines = _gate_lines(path)
    prep_end = 0
    while prep_end < len(gate_lines) and gate_name(gate_lines[prep_end]) == "x":
        prep_end += 1
    flip_counts: dict[int, int] = {}
    for line in gate_lines[:prep_end]:
        qubit = gate_qubits(line)[0]
        flip_counts[qubit] = flip_counts.get(qubit, 0) + 1
    state_prep = tuple(sorted(q for q, count in flip_counts.items() if count % 2 == 1))
    return state_prep, gate_lines[prep_end:]


def parse_lsh_qasm(
    path: str | Path,
    *,
    num_qubits: int = 120,
    rz_per_layer: int = 536,
    max_layers: int | None = None,
) -> ParsedLshCircuit:
    """Parse a hadron-dynamics QASM file into state prep plus permutation-tracked layers.

    The body is cut into layers after every ``rz_per_layer``-th ``rz`` gate. A ``swap`` updates
    a running physical -> logical permutation and is not emitted as a gate; every other gate is
    translated through that permutation before being recorded.

    Args:
        path: Path to the QASM file.
        num_qubits: Circuit width (120 for the published circuits).
        rz_per_layer: Number of ``rz`` gates per Trotter layer (536 for the published circuits).
        max_layers: Stop after this many layers instead of consuming the whole body. ``None``
            parses every layer.

    Returns:
        The parsed circuit.

    Raises:
        ValueError: If the body does not end exactly on a layer boundary (given ``max_layers``),
            or the leading state-prep run is not all ``x`` gates.
    """
    state_prep, body_lines = split_state_prep(path)

    permutation = list(range(num_qubits))
    layers: list[tuple[LogicalGate, ...]] = []
    current_layer: list[LogicalGate] = []
    rz_count_in_layer = 0
    for line in body_lines:
        if max_layers is not None and len(layers) >= max_layers:
            break
        name = gate_name(line)
        physical_qubits = gate_qubits(line)
        if name == "swap":
            a, b = physical_qubits
            permutation[a], permutation[b] = permutation[b], permutation[a]
            continue
        logical_qubits = tuple(permutation[q] for q in physical_qubits)
        angle = float(_RZ_ANGLE_RE.match(line).group(1)) if name == "rz" else None
        current_layer.append(LogicalGate(name, logical_qubits, angle))
        if name == "rz":
            rz_count_in_layer += 1
            if rz_count_in_layer % rz_per_layer == 0:
                layers.append(tuple(current_layer))
                current_layer = []
                rz_count_in_layer = 0

    if current_layer and (max_layers is None or len(layers) < max_layers):
        raise ValueError(
            f"{path}: {len(current_layer)} trailing gates after the last full layer; "
            "the body does not end on a layer boundary."
        )

    return ParsedLshCircuit(
        num_qubits=num_qubits,
        state_prep=state_prep,
        layers=tuple(layers),
        final_permutation=tuple(permutation),
    )
