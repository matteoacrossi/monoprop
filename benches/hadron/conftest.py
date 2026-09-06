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

"""Fixtures for benches/hadron's own tests.

Run with ``--confcutdir=benches/hadron`` so pytest does not also pick up ``benches/conftest.py``
(monoprop's own Bencher-tracked suite fixtures, irrelevant here and requiring ``bench_comm``).
Run ``./fetch_fixtures.sh`` first to populate ``.cache/`` with the external QASM circuits and
reference h5 data; tests that need them skip cleanly if it is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_HADRON_DIR = Path(__file__).parent
_CIRCUITS_DIR = (
    _HADRON_DIR
    / ".cache/qat/data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh"
)
_H5_PATH = _HADRON_DIR / ".cache/lsh_data/data_x100.h5"


@pytest.fixture(scope="session")
def circuits_dir() -> Path:
    """Directory holding ``x_100_SCV.qasm`` and ``x_100_meson.qasm``."""
    if not (_CIRCUITS_DIR / "x_100_SCV.qasm").exists():
        pytest.skip(f"fixtures missing; run {_HADRON_DIR / 'fetch_fixtures.sh'} first")
    return _CIRCUITS_DIR


@pytest.fixture(scope="session")
def h5_path() -> Path:
    """Path to the raw per-site reference data (``data_x100.h5``)."""
    if not _H5_PATH.exists():
        pytest.skip(f"fixtures missing; run {_HADRON_DIR / 'fetch_fixtures.sh'} first")
    return _H5_PATH
