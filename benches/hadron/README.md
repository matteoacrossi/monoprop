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

Add `--two-qubit-error 2e-3` to switch on H3's depolarizing damping, and compare the result
against the `QPU` column rather than the classical ones. `--two-qubit-error 0` is its noiseless
control: the same layer-by-layer propagation with the channel switched off.

## Layout

- `qasm_frontend.py` -- OPENQASM 2.0 parsing: state-prep split, gate histograms, and
  physical -> logical permutation tracking for the Trotter-layer segmentation checks.
- `pauli_frontend.py` -- reduces the circuit's Clifford (`h`/`cx`/`x`/`swap`) + `rz` gates into
  pure Pauli-rotation `ExpGate`s that `monoprop.Circuit` accepts (T1's front-end decision: see
  the module docstring for why `monoprop.from_qiskit_circuit` doesn't work here).
- `observable.py` -- the `n_f(t)` observable, per the paper's actual per-site occupation-number
  formula (see its module docstring for why the plan's original linear formula was wrong, and
  why `ref_x*_perqubit_diff.csv`'s 120 columns are 60 site values each duplicated across a pair).
- `noise.py` -- H3's depolarizing channel as a per-layer coefficient rescale (monoprop has no
  channel in its API), including the frame correction needed to weigh a term by its *physical*
  Pauli weight rather than its weight in the reduced frame.
- `propagate.py` -- the driver: parses a circuit prefix, builds the reduced `monoprop.Circuit`,
  and propagates `n_f(t)` through it, in one shot or layer by layer with damping.
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
- **T6** (noise-aware propagation, H3): tested, results below. Its acceptance criterion is
  **not** met, though the hypothesis is directionally confirmed.
- **T4b, T5** and the secondary Pauli-vs-Majorana comparison (plan sections 6 and 8): not
  started, but see the note on the conserved charge below, which gives T5 a reference-free
  error axis it did not have.

## T6 / H3: does noise make Pauli propagation easier?

All runs at `cutoff=20, lower_atol=1e-5`, with the noiseless control on the *same*
layer-by-layer path (`--two-qubit-error 0`), so only the channel differs. `p2q = 2e-3` is
section 4's hardware rate, not a fit; it works out to a per-qubit depolarizing probability of
`p ~ 0.028`-`0.031` per layer.

| step | noiseless | noisy 2e-3 | QPU | TN | peak terms noisy/noiseless |
|---|---|---|---|---|---|
| 1 | +1.82534 | +1.77394 | +1.77264 | +1.82422 | 1.000 |
| 2 | +1.34729 | +1.27692 | +1.23472 | +1.34382 | 0.982 |
| 3 | +0.69026 | +0.65737 | +0.70055 | +0.68530 | 0.951 |
| 4 | +0.02108 | +0.06924 | +0.11258 | +0.01782 | 0.876 |
| 5 | -0.49887 | -0.36273 | -0.35510 | -0.49577 | 0.808 |
| 6 | -0.75823 | -0.57168 | -0.47929 | -0.74474 | 0.774 |
| 7 | -0.72573 | -0.55758 | -0.40584 | -0.69942 | 0.688 |
| 8 | -0.45365 | -0.37670 | -0.31604 | -0.41497 | 0.648 |

**Cost: confirmed, but a constant factor rather than a change in scaling.** The term-count
saving grows monotonically with depth, reaching 35% in peak terms (27% in total) by step 8, and
65% at `p2q = 5e-3`. What it is *not* is a change in how cost scales: sweeping `lower_atol` at
6 layers, both configurations converge at the same rate, the damped one just needs 75-87% of
the terms for the same relative accuracy. So "simulating the noisy circuit is the easier
problem" holds, mildly. Two caveats: wall-clock does not improve, because `noise.damped` walks
every term in Python and eats the saving (the C++ engine could apply the channel *before* the
keep/drop decision and prune during evolution instead); and the channel here sits at layer
boundaries only, so intra-layer branching is undamped, making these numbers a lower bound.

**Accuracy against the QPU: better, but not within shot noise.** Mean deviation from the QPU
curve over steps 1-8 falls from 0.1434 (noiseless) to 0.0553 at `2e-3`, and to 0.0351 at
`3e-3`, which is where the residual is minimised:

| p2q | mean abs dev | max abs dev | rms |
|---|---|---|---|
| 0 | 0.1434 | 0.3199 | 0.1745 |
| 2e-3 | 0.0553 | 0.1517 | 0.0714 |
| 3e-3 | 0.0351 | 0.0799 | 0.0422 |
| 5e-3 | 0.0721 | 0.1413 | 0.0827 |

That the residual has a clear minimum at `3e-3` -- within 1.5x of the independently quoted
hardware rate -- says the channel's functional form is roughly right and the rate is off by
about half. But T6 asks for agreement *within shot noise*, and the only published error bar
(`Q_error_bar` at step 20) is 0.0057, an order of magnitude below even the best-fit residual.
The honest verdict: weight-dependent depolarizing explains most of the QPU's decay and lands
inside the plan's cross-method spread (0.0909), but it is not the whole story.

Step 3 is the clearest evidence of that: the QPU value there (+0.70055) sits *above* every
classical method, so no amount of damping can reach it -- any channel moves the prediction the
wrong way. A single depolarizing rate cannot produce that, which is a point against reading the
QPU residual as pure decay (relevant to H2).

**Not a test of H3:** a *global* depolarizing channel would give exactly
`n_f_noisy(t) = (1-q)^t n_f(t)`, which is section 4's fitted `F(t)` -- reproducible with no
simulation at all, and with no cost saving, since a uniform rescale does not preferentially
damp the terms that branch. Only the weight-dependent channel is informative. See
`noise.py`'s module docstring.

### A conserved charge, for free

`60 - sum_r [<Z_i(r)> + <Z_o(r)>]/2` is just the total occupation `sum_j n(j)`, and every one
of the 120 wires is propagated anyway, so it costs nothing to read off (`Q(SCV)`/`Q(meson)` in
`example.py`'s output). It is conserved by these dynamics to float precision -- 7e-15 with
nothing truncated. Individual reduced Pauli rotations do *not* conserve it (a lone `XX`
anticommutes with `Z_1 + Z_2`); it is restored at each layer boundary, which is where the
measurement and the channel sit.

**It is a correctness invariant, not an error estimator.** That distinction took measuring to
establish. At 1-3 layers the charge stays exact to 1e-14--1e-11 *even under a budget tight
enough to put ~1e-3 of error on `n_f`*, because the contributions that cancel across wires
carry equal-magnitude coefficients and a magnitude threshold keeps or drops both. The symmetry
is largely protected from the truncation rule. At 6 layers it does drift, but not usefully
monotonically:

| lower_atol | charge drift | n_f error |
|---|---|---|
| 1e-3 | 3.07e-03 | 7.77e-03 |
| 3e-4 | 1.63e-04 | 1.40e-03 |
| 1e-4 | 8.23e-06 | 9.80e-04 |
| 3e-5 | 5.09e-04 | 9.29e-04 |
| 1e-5 | 3.30e-05 | 3.20e-04 |

So it is a free smoke alarm and a strong end-to-end check on the site mapping, the conjugation
frames, the signs and the channel -- any of which would break it -- but it is no substitute for
a convergence study, and T5 still needs one.

It survives the channel too: both initial states are exactly half filled, so `sum_j <Z_j> = 0`
at `t = 0`, and depolarizing scales that zero sum to zero. Handy here, but a property of this
half-filled instance rather than a general guarantee.

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
