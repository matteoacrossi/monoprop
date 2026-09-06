# Plan: reproduce and stress the SU(2) LSH hadron-dynamics classical baseline

Working notes for a Claude Code session in the `monoprop` repo. Read this whole
file before writing code.

"Established facts" were verified directly against the published QASM and data;
treat them as ground truth and re-derive them as smoke tests. "Hypotheses" are
unverified and are what the work is meant to test. Do not conflate the two.

**Start small.** The first milestone is 3 Trotter steps, not 20. See §6.

---

## 1. Background

Paper: Ilčić, Majumdar, Mathew, Ali, Earnest-Noble, Raychowdhury,
*Observation of Robust and Coherent Non-Abelian Hadron Dynamics on Noisy Quantum
Processors*, arXiv:2602.18080 (v3).

(1+1)D SU(2) lattice gauge theory in the Loop-String-Hadron encoding, 60
staggered sites on 120 qubits of an IBM Heron processor, 20-25 Trotter steps,
benchmarked against tensor networks (TN) and Pauli propagation (PP) on CPU and
GPU. They report the classical baselines becoming expensive while QPU runtime
stays fixed by the shot budget.

The paper's code is not public and the Methods/Supplementary sections are not
retrievable from the arXiv HTML render. **We do not need them.** The compiled
circuits are published as QASM and the PP_GPU per-qubit results as CSV.

> **Do not derive the LSH Hamiltonian.** The QASM is the specification. If you
> find yourself reasoning about LSH ladder operators or prepotentials, stop.

### Goals

1. **Reproduce** the published PP_GPU per-qubit results with `monoprop`'s
   `PauliPropagator`, starting at 3 Trotter steps.
2. **Test whether the reported classical cost is real** (§5, H1).
3. **Test whether the QPU adds information** beyond a decaying multiple of the
   classical answer (§5, H2).
4. **Secondary:** Pauli vs Majorana truncation ordering (§8). Only after
   milestone T4b passes.

---

## 2. Fixtures

### Circuits

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io.git qat
cd qat && git sparse-checkout set \
  data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh
```

| file | size | contents |
|---|---|---|
| `x_100_SCV.qasm` | 481029 B | strong-coupling-vacuum initial state |
| `x_100_meson.qasm` | 481047 B | meson-on-SCV initial state |
| `README.md` | 5806 B | parameters, runtimes, 20-step benchmark table |

### Reference data

```bash
git clone --depth 1 https://github.com/mathew0036/lsh_data.git
```

Contains `data_x{50,80,100,200}.h5` and `paulipropGPU_data/` with
`results_{general,mid}_x{50,100,200}_gpu.csv`. Each CSV is 20 rows x (1 step
index + 120 per-qubit `<Z_j>`). `general` = SCV, `mid` = meson.

Shipped alongside this plan, derived from those CSVs:

- `lsh_x100_benchmark.csv` — the 20-step scalar table (TN, PP_CPU, PP_GPU, QPU,
  median, stdev)
- `ref_x{50,100,200}_perqubit_diff.csv` — **primary regression fixture**:
  per-qubit differential `<Z_j>_meson - <Z_j>_SCV` for all 20 steps, plus the
  reconstructed `n_f` column

### Instance parameters

- `N = 60`, `x = 100`, 120 qubits, `dt = 0.0015`, first-order Trotter, 20 steps
- 10,000 shots/step; two-qubit depth 259 at step 20
- PP_GPU truncation: coefficient threshold `1e-5`.
  PP_CPU: cap on retained terms, calibrated by extrapolation from small systems.
- TN: 2-site TDVP, `D_max = 200`, `2j_max = 5`, no Trotter error
- reported step-20 runtimes: QPU 20 s, TN 2484.37 s, PP_CPU 2930.44 s,
  PP_GPU 8949.11 s

> **Term counts are not published anywhere** — not in the main text, the tracker
> README, or the data repo. Only the truncation *type* is stated. So we cannot
> match their budget; we must report our own and compare on accuracy achieved,
> not on threshold value. See T5.

---

## 3. Established facts: the circuits

**Gate inventory** (identical in both files):

| gate | count |
|---|---|
| `cx` | 11840 |
| `rz` | 10720 |
| `h` | 4720 |
| `swap` | 2330 |
| `x` | 2460 (SCV) / 2462 (meson) |

Two-qubit total = 11840 + 3x2330 = 18830, consistent with the paper's ">17000".

**The two circuits differ by exactly two gates.** Multiset diff of non-empty
lines returns `x q[59];` and `x q[60];` extra in the meson file, nothing else.
The Trotter body is byte-identical.

Consequence: back-propagate the observable through the body **once**, then
evaluate against two product states. Do not run two propagations. Any
depth-dependent global damping also cancels exactly in the differential.

**State prep is the first 62 gates** (all `x`); the body is the remaining 32010.

**The body is exactly 20 identical Trotter layers.** Segmentation rule: cut
after every 536th `rz`. This yields 20 blocks; with `swap` lines removed every
block is **byte-identical** at 1484 gates (`cx`/`h`/`rz`).

**Routing:** block 0 carries 88 swaps composing to a non-trivial permutation;
blocks 1-19 carry 118 swaps each which compose to the identity *within* the
block. Cumulative permutation over the body is not the identity.

> Do **not** simply delete the `swap` lines. Even where they cancel within a
> block, gates between them address physical positions, so the swap-free gate
> list is not the logical layer. Parse sequentially maintaining a
> physical->logical permutation: a `swap` updates the permutation and emits
> nothing; `cx`/`h`/`rz` act on whichever logical qubits currently sit at those
> physical positions.

**Rotation angles — only six distinct values, all non-Clifford:**

| angle | count | `sin(theta)` |
|---|---|---|
| `+0.15` | 2360 | 0.14944 |
| `-0.15` | 2360 | 0.14944 |
| `+0.03` | 1200 | 0.03000 |
| `-0.03` | 1200 | 0.03000 |
| `+0.005` | 1200 | 0.00500 |
| `-0.005` | 2400 | 0.00500 |

`0.15 = x*dt = 100*0.0015` (interaction term); `0.03` is the mass term at
`m/g = 1`. Origin of `0.005` unidentified — worth pinning down, not blocking.

---

## 4. Established facts: observable, layout, and the QPU decay

### Observable and qubit layout — exact, verified

$$n_f(t) = -\frac{1}{4}\sum_{j=0}^{119} (-1)^{\lfloor j/2 \rfloor}
\left[\langle Z_j\rangle_{\text{meson}} - \langle Z_j\rangle_{\text{SCV}}\right]$$

This reproduces all 20 published PP_GPU values to `1.1e-16` — machine
precision, so it is the exact convention rather than a fit.

The `floor(j/2)` means **sites are interleaved pairs**: `q[2r]` and `q[2r+1]`
hold the two LSH quantum numbers of site `r`. (An earlier draft of this plan
guessed two 60-qubit registers; that was wrong.)

> **Caveat.** The formula above is verified in the *logical* indexing of the
> PP_GPU CSVs. The QASM indices are physical and need not agree, since there are
> 2330 routing swaps. Reconciling CSV logical order with QASM physical order is
> a real task (T3), not an assumption. The state-prep pattern
> (`x q[1]; x q[61]; x q[3]; x q[63]; ...`) does not obviously match interleaved
> pairs, which is exactly why the permutation must be tracked.

### QPU amplitude decay

Fitting `QPU(t)/classical(t)` where `|classical| > 0.30` and signs agree
(13 of 20 steps) to `F(t) = exp(-lambda*t)`, bootstrap errors, 2000 resamples:

| reference | lambda /step | R^2 | F(1) |
|---|---|---|---|
| TN | 0.059 +/- 0.014 | 0.75 | 0.964 |
| PP_GPU | 0.058 +/- 0.014 | 0.75 | 0.965 |
| PP_CPU | 0.072 +/- 0.017 | 0.61 | 1.040 |

Zero crossings must be excluded or the ratio diverges. `lambda = 0.059/step`
with per-2q-gate error `p ~ 2e-3` implies ~30 effective error locations per
step, unremarkable for this circuit.

At step 20 the QPU gives `n_f = -0.0772` against TN `+0.2778`, PP_CPU `+0.4082`,
PP_GPU `+0.2456`. The QPU signal has decayed through zero and changed sign.
Step 20 is the point submitted to the tracker as the reported observable.

---

## 5. Hypotheses

- **H1 (cheap PP).** Truncated PP reproduces the published curve at a budget far
  below what their runtimes imply — target minutes, single process, no MPI.
  *Rationale:* each branching costs `sin(theta)`; at `1e-5` only ~6 branchings
  of the dominant `0.15` angle survive (`0.14944^6 = 1.11e-5`), and the smaller
  angles are suppressed much harder. Estimated depth by tolerance:

  | tolerance | max branchings at 0.15 | at 0.03 | at 0.005 |
  |---|---|---|---|
  | 1e-3 | 3 | 1 | 1 |
  | 1e-4 | 4 | 2 | 1 |
  | 1e-5 | 6 | 3 | 2 |

  Term count scales as `C(B,k)` with `B` = branching opportunities on the
  support. For `B ~ 100`: `k=3` -> 1.6e5 terms (13 MB at 80 B/term);
  `k=4` -> 3.9e6 (0.3 GB); `k=5` -> 7.5e7 (6 GB); `k=6` -> 1.2e9 (95 GB).
  `B` is the one parameter not derivable from the file — measure it.
  *Counter-consideration:* the combinatorial factor genuinely fights the small
  angle. `C(B,k)` also overcounts, since distinct paths often reach the same
  Pauli string and merge, and a rotation only branches if it anticommutes.
  Do not assume H1; measure error vs budget.

- **H2 (no independent QPU information).** After dividing out the fitted `F(t)`,
  the QPU residual is consistent with 10k-shot noise and shows no systematic
  structure the classical methods lack.

- **H3 (noise makes PP easier).** A per-layer depolarizing channel — rescale
  each term's coefficient by `(1-p)^w` in Pauli weight `w` — *reduces* cost and
  makes truncation a controlled approximation. Simulating the noisy circuit is
  the easier problem; their PP baseline simulated the noiseless one.
  *Limitation to state in any writeup:* T1 amplitude damping is non-unital and
  does not reduce to weight-dependent Pauli damping; coherent errors do not damp
  at all.

---

## 6. Task order

Sequential. Each task has an acceptance criterion; do not proceed without it.

### T1 — Decide the front-end strategy (before writing code)

`monoprop` 0.9.0 exposes: `Circuit`, `ExpGate`, `PauliOperator`, `Pauli`,
`PauliPropagator`, `MajoranaOperator`, `MajoranaPropagator`,
`MonomialPropagator`, `FermiOperator`, `FermiString`,
`jordan_wigner_basis_change`, `conversion_utils`, `validate_parameter_mapping`,
`has_mpi`.

Options: **(a)** feed the Clifford+Rz sequence directly if the API takes
Clifford gates as first-class objects; **(b)** recompose each `cx`-ladder /
`h`-conjugated `rz` into `ExpGate(PauliOperator(...))` generators.

Prefer (a) — less code, fewer transcription bugs. Inspect the API and
`docs/notebooks/` before committing. Fall back to (b) only if (a) is
unsupported.

**Acceptance:** a note in the PR description stating which path and why.

### T2 — QASM front end

Parse OPENQASM 2.0, `qreg q[120]`, gates `x`/`h`/`cx`/`swap`/`rz(theta)`.
Split off the 62 state-prep `x` gates. Segment the body by every 536th `rz`.
Track the physical->logical permutation as described in §3.

**Acceptance, in order:**

1. Gate histogram matches §3 exactly for both files.
2. Multiset diff of parsed bodies is empty; the only difference between the two
   files is the two state-prep X gates.
3. **All 20 logical layers are identical after permutation tracking.** This is
   the strong test — it catches routing-handling bugs immediately.

### T3 — Observable and layout reconciliation

Establish the map between QASM physical indices and the CSV logical indices in
which §4's formula holds.

Sanity checks: SCV gives `n_f = 0` at every site at `t = 0`; the meson state
gives total `n_f = 2`; the published curve starts at `n_f(1) = 1.8253`, decaying
from 2.

**Acceptance:** both initial states reproduce their `t=0` values exactly, and
`ref_x100_perqubit_diff.csv` row 1 is reproduced qubit-by-qubit at `t=1`.

### T4a — THREE STEPS (first real milestone)

Propagate the differential observable through 3 Trotter layers. Start at
tolerance `1e-3` (k<=3, ~1e5 terms, ~13 MB — trivially laptop-scale).

**Acceptance:** per-qubit agreement with rows 1-3 of
`ref_x100_perqubit_diff.csv` to better than `1e-3` on every one of the 120
qubits, and `n_f(1..3)` matching `+1.825336`, `+1.347871`, `+0.693895`.

Log wall-clock time and retained term count per step. Stop and fix before
going further — a layout or routing error will show up here as a heatmap that
is wrong in a *structured* way, which is far easier to diagnose at 3 steps than
at 20.

### T4b — Ten steps

Same, steps 1-10. These are cheap at any tolerance since cost grows with depth.

**Acceptance:** per-qubit agreement to a few times `1e-3` across steps 1-10,
plus a logged curve of term count and wall time vs step. That curve extrapolates
to step 20 and answers empirically whether 20 steps needs a bigger machine.

### T5 — Error vs budget (H1)

Sweep tolerance and/or max-term cap; plot error against the reference and
against wall time.

**Acceptance:** report the tolerance needed to land inside their *cross-method
spread* (mean stdev across the 20 steps is 0.0909; at step 20 it is 0.2062),
not the tolerance needed to match `1e-5`. Since their term counts are
unpublished, accuracy-per-unit-cost is the only honest comparison.

### T6 — Noise-aware propagation (H3)

Check whether a Pauli/depolarizing channel exists in the API. If not, it is a
coefficient rescale by `(1-p)^w` after each layer — a small addition. Use `p`
fitted from §4, not a guessed value.

**Acceptance:** noisy PP reproduces the *QPU* curve (not the classical one)
within shot noise, at a lower budget than noiseless PP needs.

---

## 7. Deliverable shape

A benchmark under `benches/`, not a change to the core library. Nothing in
`include/monoprop` or `src` should need touching for T1-T5; T6 may add a channel.

Repo conventions: `AGENTS.md` for workflow, `CONTRIBUTING.md` before a PR (CLA
required). Per `README.md`, any PR changing behaviour or public API must update
`AGENTS.md`, `README.md` and `docs/` in the same change. Build with the CI
presets (`release-gcc`); tests via `uv run python -m pytest -m "not mpi"` and
`ctest --preset release-gcc`. MPI is off by default and is not needed here —
do not enable it.

---

## 8. Secondary: Pauli vs Majorana truncation

Only after T4b passes.

Jordan-Wigner is a bijection between Pauli strings and Majorana monomials, so
untruncated the two propagators produce identical branching trees term for term.
Only the discard ordering differs, because weight differs: `X_j X_{j+1}` is
Pauli weight 2 and Majorana weight 2, while `Z_j Z_{j+1}` is Pauli weight 2 and
Majorana weight 4. Diagonal density-like terms are penalised under Majorana
truncation; hopping-like terms are favoured.

60 sites = 120 fermionic modes = 240 Majoranas. There is no compression from
working "in fermionic space" — the qubit count already equals the mode count.

Two diagnostics, in order:

1. **Weight histogram.** Count how many parsed generators are
   Majorana-quadratic. Quadratic generators are Gaussian and do not branch at
   all in the Majorana basis. A meaningful quadratic fraction is a much stronger
   reason to expect an advantage than the ordering argument, and costs nothing
   to check.
2. **Matched-budget comparison** at `N = 8-12` where exact dense evolution gives
   ground truth. Same circuit, same budget, error vs budget for both
   propagators.

Before trusting any comparison, verify (i) `cutoff` means the same thing in both
propagators — term count, weight, or coefficient threshold — and (ii) the JW
convention is identical on both sides. Round-trip a known operator through
`jordan_wigner_basis_change` and assert equality as a unit test.

---

## 9. Pitfalls

- **Do not derive the Hamiltonian.** The QASM is the specification.
- **Do not delete the `swap` lines.** Track the permutation (§3).
- **Do not assume the QASM index order matches the CSV order.** §4's formula is
  verified in logical order only.
- **Zero crossings.** `n_f(t)` oscillates through zero. Any ratio, relative
  error or fit must exclude steps near a crossing. The §4 fit is only meaningful
  because of that filter.
- **Do not treat the reported runtimes as a serious classical baseline.** PP_GPU
  (8949 s) is 3x slower than PP_CPU (2930 s), which suggests they are not
  optimised. Report our own timings on stated hardware.
- **Scope conclusions correctly.** Showing this *instance* is classically cheap
  does not show that noiseless LSH dynamics at `x = 200` is easy. What it would
  undercut is the specific claim that this run is a step toward advantage.
- **`x = 50` and `x = 200`** have per-qubit reference data shipped here but no
  QASM in the tracker directory. `x = 200` is the interesting one — TN is
  reported to break down there after step 10. Check for its QASM once T4b
  passes.
