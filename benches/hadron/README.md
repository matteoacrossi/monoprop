# SU(2) LSH hadron-dynamics reproduction

Reproduces and stress-tests the classical (Pauli propagation) baseline reported in Ilčić,
Majumdar, Mathew, Ali, Earnest-Noble, Raychowdhury, *Observation of Robust and Coherent
Non-Abelian Hadron Dynamics on Noisy Quantum Processors* (arXiv:2602.18080), using monoprop's
`PauliPropagator`. See [`PLAN_lsh_pauli_prop.md`](PLAN_lsh_pauli_prop.md) for the full task list
and background; this README covers only how to run things.

This is a research reproduction, not part of monoprop's own Bencher-tracked benchmark suite (see
[`../README.md`](../README.md)) -- it has its own dependency group and does not use
`../conftest.py`'s fixtures.

## Setup

```bash
uv sync --group hadron
./fetch_fixtures.sh   # clones the QASM circuits and reference h5 data into .cache/ (gitignored)
```

If `uv run` tries to rebuild the C++ extension and that rebuild fails (unrelated to anything in
this directory -- pure Python), add `--no-sync` to skip the rebuild and use the existing build.

## Quick example

Compare a couple of Trotter steps against the published TN/PP_CPU/PP_GPU/QPU values (needs the
`-m` form, run from the repository root, for the `benches.hadron` imports to resolve):

```bash
uv run --group hadron python -m benches.hadron.example --layers 1 2
```

## Layout

- `qasm_frontend.py` -- OPENQASM 2.0 parsing: state-prep split, gate histograms, and
  physical -> logical permutation tracking for the Trotter-layer segmentation checks.
- `pauli_frontend.py` -- reduces the circuit's Clifford (`h`/`cx`/`x`/`swap`) + `rz` gates into
  pure Pauli-rotation `ExpGate`s that `monoprop.Circuit` accepts (T1's front-end decision: see
  the module docstring for why `monoprop.from_qiskit_circuit` doesn't work here).
- `observable.py` -- the `n_f(t)` observable, per the paper's actual per-site occupation-number
  formula (see its module docstring for why the plan's original linear formula was wrong, and
  why `ref_x*_perqubit_diff.csv`'s 120 columns are 60 site values each duplicated across a pair).
- `propagate.py` -- the driver: parses a circuit prefix, builds the reduced `monoprop.Circuit`,
  and propagates `n_f(t)` through it.
- `example.py` -- a small CLI wrapping `propagate.py`, printing `n_f(t)` next to the published
  benchmark table for a few chosen layer counts (see "Quick example" above).
- `fetch_fixtures.sh` -- reproduces the `.cache/` clone (QASM circuits + `lsh_data` h5 files).

## Running the tests

`benches/conftest.py` (monoprop's own bench suite fixtures) must not be picked up here --
`--confcutdir` stops pytest from walking up past this directory:

```bash
uv run pytest --confcutdir=benches/hadron benches/hadron
```

Tests that need the external fixtures skip cleanly (with a message pointing at
`fetch_fixtures.sh`) if `.cache/` hasn't been populated yet.

## Where the reproduction currently stands

- **T1/T2** (front-end decision, QASM parsing): done; `test_qasm_frontend.py` covers all three
  of the plan's T2 acceptance criteria.
- **T3/T4a** (observable reconciliation, 3-Trotter-layer milestone): done for the scalar
  `n_f(t)` and the true per-*site* values (60, not 120 -- see `observable.py`), validated
  against the raw `data_x100.h5` reference (exact at 1 layer, within its own truncation budget
  at 2-3 layers) in `test_milestones.py`.
- **T4b, T5, T6** and the secondary Pauli-vs-Majorana comparison (plan sections 6 and 8): not
  started.

### Open item worth flagging

The per-site formula (`observable.n_f`, `observable.site_diff`) was reverse-engineered from the
paper plus `data_x100.h5`, not from any documented qubit-routing table (the paper's Methods
section has no routing/connectivity details, and the tracker README has none either). It ended
up simple -- site `r`'s two qubits are output wires `2r` and `2r+1` (no permutation correction
needed beyond what `pauli_frontend.reduce_output_pauli` already does), with the paper's `2 - x`
flip applying to odd `r` -- and is validated **exactly** (float precision) against every one of
the 60 sites at step 1, and within `PP_GPU`'s own truncation tolerance at steps 2-3, i.e. across
60 sites x 3 depths, not a single coincidental match. Still empirically derived rather than
read off a documented convention, so it's worth a second, independent check (e.g. against
`x=50`/`x=200`'s h5 files once T4b needs them) before leaning on it for anything downstream.
