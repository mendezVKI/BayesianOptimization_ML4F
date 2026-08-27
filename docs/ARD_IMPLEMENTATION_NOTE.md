# Automatic Relevance Determination in `BO_ML4F`

**Status: IMPLEMENTED, opt-in, default off. 2026-08-26.**
Companion to `docs/BO_ML4F_documentation.tex`. Written to be read by someone deciding whether to
turn ARD on in a campaign, not only by someone maintaining the kernel code.

---

## 0. One-paragraph summary

`BO_ML4F` had a single scalar length scale `l_c` shared by every input coordinate. This note
records the addition of **Automatic Relevance Determination** — one length scale per coordinate —
as an opt-in flag, `GPConfig(ard=True)`. Nothing changes for existing callers: with a scalar `l_c`
the library takes the same code path it always did and returns the same numbers, and the test suite
asserts this with exact equality against frozen copies of the pre-ARD kernel formulas. Three
distinct usages are supported (fitted ARD, frozen ARD, input whitening), and section 6 explains why
the second and third — not the first — are the ones the RSPA campaigns should use.

---

## 1. Why

### 1.1 What the isotropic kernel assumes

The Matérn-5/2 and squared-exponential kernels in this library measure how far apart two inputs are
with a single Euclidean distance divided by one number:

```
k(x, x') = sigma_f^2 * phi( ||x - x'|| / l_c )
```

That is a statement about the objective, not about the inputs: it says the objective decorrelates at
the same rate along every coordinate. Mapping the inputs onto the unit box first — which every
caller of this library does — makes the coordinates *commensurable*, but it does not make them
*equally influential*. Bounds come from admissibility (what the actuator may be commanded to do);
the relevant scale for a GP is where the objective decorrelates. They are unrelated quantities.

### 1.2 What was measured on the RSPA Burgers case

Two independent measurements in `RSPA_Paper/Burger_CASE/RT_ML4F/experiments/` motivated this work.
They are summarised here so this note stands on its own; the numbers belong to those files.

* `12_GP_ANISOTROPY.md` — an ARD fit to 448 uniform-design plant episodes recovers a **147×** spread
  in per-coordinate length scales across the 13-D policy box, cuts held-out RMSE from 11.22 to 8.22,
  lifts prediction Spearman from 0.748 to 0.883, and reassigns **73 %** of what the isotropic kernel
  booked as observation noise (`sigma_y^2` 0.3124 → 0.0852) to structure it could not represent.
  Seven of thirteen coordinates sit on the upper bound, i.e. no variation detected over the box.
* `13_REDUCED_POLICY_SPACE.md` — in the four-coordinate core the objective actually depends on, the
  spread collapses to **5.3×** and ARD then buys nothing measurable (RMSE 8.225 isotropic against
  8.221 with ARD).

The second result is as important as the first. **ARD is worth having because it measures the
anisotropy, not because a campaign should necessarily run on it.** Section 6 develops this.

### 1.3 What inflated noise does

The failure mode is worth naming, because it is silent. Under an isotropic kernel, two points that
differ only in inert coordinates look like replicate observations of the same input. When their
observed values disagree — as they must, since they are not actually the same input — the likelihood
has nowhere to put the difference except `sigma_y`. The fitted noise inflates, the posterior mean
flattens, and the acquisition loses its ability to rank candidates. Nothing warns you: the fit
converges, the diagnostics look plausible, and the GP has quietly become a constant.

---

## 2. The mathematics

ARD replaces the scalar by a vector `l = (l_1, ..., l_d)` and the distance by

```
r_ARD(x, x')^2 = sum_i (x_i - x'_i)^2 / l_i^2
```

The single identity that governs the whole implementation is

```
k_ARD(x, x'; l, sigma_f) == k_iso(x/l, x'/l; 1, sigma_f)
```

**ARD is exactly an isotropic kernel applied to rescaled inputs.** Three consequences follow, and
all three are load-bearing:

1. The implementation is a coordinate rescaling, not a new kernel. Both kernels are implemented by
   dividing the inputs and then calling the existing isotropic routine.
2. Anything that can rescale its inputs before handing them to this library gets ARD without the
   library needing to know — this is the "frozen whitening" route of section 6.2.
3. A trust region that is a sphere of radius `h` in the rescaled metric is an **ellipsoid** with
   semi-axes `h * l_i` in the original coordinates. Any caller that derives a search radius from
   `l_c` must be rewritten per coordinate when ARD is enabled. This is not optional; see 6.4.

The hyperparameter vector optimised by HPO becomes

```
theta_log = log([l_1, ..., l_d, sigma_f, sigma_y])      # d + 2 entries
```

which for `d = 1` is exactly the historical 3-entry layout. That is why the layout can be inferred
from the vector's length and no extra argument was needed anywhere.

---

## 3. What changed, function by function

All changes are in `BO_ML4F.py`. Nothing in `BO_func_YL.py` was touched.

| # | Location | Change |
|---|---|---|
| 1 | `GPConfig` | `l_c` typed `Union[float, Array]`; new fields `ard: bool = False` and `hpo_random_state: Optional[int] = None`; `theta0_log` / `theta_bounds_log` documented as broadcastable |
| 2 | new helpers | `as_length_scales`, `is_ard`, `format_length_scales`, `ard_whitening_weights`, and private `_expand_theta_bounds`, `_expand_theta_vector`, `_split_theta`, `_pack_length_scales` |
| 3 | `rbf_kernel_amp` | scalar → historical path verbatim; vector → divide inputs by `l`, then the same `rbf_kernel_` call with `gamma = 0.5` |
| 4 | `matern52_kernel_amp` | scalar → historical path verbatim; vector → ARD-scaled distance |
| 5 | `kernel_amp`, `gp_fit`, `gp_predict` | signatures widened, docstrings only — they pass `l_c` straight through |
| 6 | `GPModel` | `l_c` typed `Union[float, Array]`; `theta_log` layout documented |
| 7 | `negative_log_marginal_likelihood` | unpacks `theta_log` by length via `_split_theta`; `param_hist` records every length scale (still 3 entries when isotropic) |
| 8 | `optimize_gp_hyperparams` | optimises `n_ell + 2` parameters, where a **vector `l_c` implies ARD even when `ard=False`** (refitting one scalar over an anisotropic seed would silently discard the declared anisotropy); broadcasts the HPO box and `theta0_log`; **discards a warm start of the wrong length** instead of reshaping it; seeds restarts from `hpo_random_state`; clips restarts into the box when one is declared |
| 9 | `build_gp_model` | copies a vector `l_c` rather than referencing the config's array |
| 10 | `fit_gp` | resolves the length-scale layout against `X.shape[1]`: expands a scalar seed to `d` entries when `ard=True`, validates a vector against `d` |
| 11 | `estimate_ard_length_scales` | **new public function** — the offline screening entry point (section 6.1) |
| 12 | `_export_iteration_data` | `l_c` always exported as a 1-D array: `(1,)` isotropic, `(d,)` ARD |
| 13 | `bayesian_optimization` | history stores a copy of a vector `l_c`; the log line renders it through `format_length_scales` |

`update_gp_rank1` needed **no change**: it routes through `kernel_amp`, and `k(x, x) = sigma_f^2`
regardless of the length scales. The test suite proves the rank-1 extension still reproduces a full
refit under ARD.

### 3.1 Why the scalar path is duplicated rather than unified

Both kernels branch on `ell.size == 1` and keep the original arithmetic verbatim in that branch,
even though the ARD branch would produce the same answer to ~1e-16. This is deliberate. The
isotropic path is what every archived RSPA result was produced with, and a floating-point difference
in the kernel propagates through Cholesky, HPO and acquisition into a *different sequence of
sampled points*. Bit-identity is cheap to keep here and expensive to argue about later. The tests
assert it with `assert_array_equal`, not `allclose`.

---

## 4. API and compatibility contract

**Guaranteed unchanged** when `ard` is left at its default `False` and `l_c` is a scalar:

* every kernel value, bit for bit (`tests/test_ard.py::test_scalar_length_scale_is_bit_identical_to_legacy`);
* `theta_log` layout and length (3);
* the `param_hist` record layout (3 entries);
* `gp.l_c` remains a Python `float`, so `f"{gp.l_c:.3e}"` in downstream code keeps working;
* the exported `.npz` field `l_c` remains a length-1 array, as before;
* HPO restarts remain unseeded unless `hpo_random_state` is set.

**New surface:**

```python
bo.GPConfig(ard=True)                      # fit one length scale per coordinate
bo.GPConfig(l_c=np.array([...]))           # frozen ARD; no HPO needed
bo.GPConfig(hpo_random_state=7)            # reproducible HPO restarts

bo.estimate_ard_length_scales(X, y, ...)   # offline screening -> dict
bo.as_length_scales(l_c, d)                # validate / normalise a specification
bo.is_ard(l_c)                             # branch downstream code safely
bo.format_length_scales(l_c)               # logging
bo.ard_whitening_weights(l_c)              # 1/l, unit geometric mean
```

**Broadcasting rule.** `theta_bounds_log` and `theta0_log` may be given with 3 entries even in ARD
mode; the length-scale entry is then applied to all `d` coordinates. This is what allows an existing
isotropic configuration — including `RT_ML4F`'s `PolicySearchConfig.gp_theta_bounds_log`, which
validates a 3-tuple — to be switched to ARD **without editing its HPO contract**.

**One asymmetry worth knowing.** The `ard` flag only decides whether a *scalar* `l_c` is expanded
to `d` length scales. A *vector* `l_c` is anisotropic whatever the flag says, and HPO will optimise
`d + 2` hyperparameters over it rather than collapsing it back to one number. This is what makes
`GPConfig(l_c=frozen_scales, optimize_hyperparams=True)` behave the way a reader expects.

**One behaviour that is not backwards compatible by construction:** a `GPModel` carried across a
change of `ard` will have a `theta_log` of the wrong length. It is discarded, not reshaped, and HPO
restarts from the configured seed. This is the safe direction — reshaping would silently invent
hyperparameters — but it means the first fit after such a switch has no warm start.

---

## 5. Cost

Let `n` be the number of observations and `d` the input dimension.

* **Per NLL evaluation:** unchanged, `O(n^2 d)` to build `K` plus `O(n^3)` for the Cholesky. ARD
  adds a divide over the inputs and nothing else.
* **Per HPO call:** L-BFGS-B has no analytic gradient here, so SciPy finite-differences it. The
  gradient costs `d + 2` NLL evaluations instead of 3 — i.e. HPO gets roughly `(d+2)/3` times more
  expensive per iteration, plus more iterations to converge in a higher-dimensional space.
* **Measured**, 13-D, `n = 55`, Matérn-5/2, 3 restarts, one HPO call: **0.10 s isotropic against
  0.37 s with ARD**. At the sample sizes this library is used at, the cost is not the objection.

The objection is statistical, not computational, and it is section 6.3.

---

## 6. How to use it

### 6.1 Fitted ARD, offline — the screening use (recommended)

This is what ARD is unambiguously good for: measuring the anisotropy of an objective from an
archive of evaluations, once, with `n` large.

```python
import numpy as np, BO_ML4F as bo

result = bo.estimate_ard_length_scales(
    X_unit, y,                       # (n, d) in the unit box, (n,) observations
    kernel="matern52",
    theta_bounds_log=[(np.log(0.05), np.log(20.0)),   # length scales (broadcast)
                      (np.log(0.20), np.log(5.00)),   # sigma_f
                      (np.log(0.02), np.log(1.50))],  # sigma_y
    n_restarts=8,
    random_state=0,                  # set it: the fit is then reproducible
)

print(result["length_scales"])       # (d,)
print(result["spread"])              # max(l)/min(l) -- the headline number
print(result["relevance_order"])     # coordinate indices, most relevant first
print(result["whitening_weights"])   # 1/l, unit geometric mean
```

**Read the output carefully.** A length scale sitting on the upper bound of the box is *not* a
measurement — it means "no variation detected across the box at this sample size". Report it as
such. And ARD relevance is the *scale of variation*, not variance explained: the coordinate with the
shortest length scale is not automatically the one whose removal costs most. Those are different
questions and must not be conflated.

### 6.2 Frozen ARD, in-campaign — the pre-registered route (recommended)

Take the length scales measured in 6.1, declare them before the campaign, and never re-fit them.

```python
gp_cfg = bo.GPConfig(
    l_c=frozen_length_scales,        # (d,) array, declared in a contracts file
    sigma_f=..., sigma_y=...,
    kernel="matern52",
    normalize_y=True,
    optimize_hyperparams=False,      # or True with ard=False to keep fitting sigma_f/sigma_y only
)
```

Equivalently, and identically in exact arithmetic (the identity of section 2), rescale the inputs
and keep a scalar kernel:

```python
w = bo.ard_whitening_weights(frozen_length_scales)
gp_cfg = bo.GPConfig(l_c=1.0, ...)   # fit on X * w, predict on X_query * w
```

`tests/test_ard.py::test_frozen_ard_predicts_as_well_as_fitted_ard` asserts these two formulations
agree to 1e-9 on both posterior mean and variance.

Why this route and not 6.3: the weights are declared before the campaign and archived with it, so
they cannot drift between arms of a paired comparison. Re-fitting length scales *during* a campaign
makes the metric campaign-adaptive, which is exactly the objection raised against an adaptive trust
scale in `8_STEP_6_BRIEF.md` §3.1.

### 6.3 Fitted ARD, in-campaign — supported, but think first

`GPConfig(ard=True, optimize_hyperparams=True)` works and is tested. It is nonetheless the option to
justify rather than assume, for two measured reasons.

**Reason one — it spends observations you are also searching with.** From
`14_STATUS_2026-08-25.md` §1.4, held-out Spearman on the Burgers policy objective, 20 subsamples:

| n | 13-D isotropic | 13-D ARD | 4-D core, isotropic |
|---:|---|---|---|
| 30 | 0.464 ± 0.13 | 0.575 ± 0.18 | **0.704 ± 0.06** |
| 55 | 0.573 ± 0.06 | 0.703 ± 0.11 | **0.785 ± 0.03** |
| 110 | 0.655 ± 0.04 | 0.800 ± 0.06 | **0.832 ± 0.03** |

ARD beats isotropic on the mean at `n >= 55`, and loses to a dimensionality reduction derived
*beforehand* at every budget. At `n = 30` it is worse than isotropic on RMSE — twelve extra
hyperparameters fitted from thirty points is overfitting, and it shows.

**Reason two — the spread column.** ARD varies ±0.11 between campaigns against ±0.06 isotropic and
±0.03 reduced. When the claim a study is making is about reproducibility, a kernel whose behaviour
swings between repeats works directly against it.

**A demonstration of the mechanism** (synthetic, 13-D, `n = 55`, two active coordinates, HPO box
`l in [0.55, 5.0]` — the box `RT_ML4F` currently declares):

```
ard=False   nll = 77.53   n_theta =  3   sigma_y = 0.020
ard=True    nll = 19.12   n_theta = 15   sigma_y = 0.147
   l = [0.55  5  5  5  2.05  5  5  5  5  5  5  5  5]
```

ARD finds both active coordinates and pins the other eleven at the ceiling, for 58 nats. But note
what the active coordinate did: **it went to the lower bound.** A floor of 0.55 chosen to stop an
*isotropic* kernel collapsing to the diagonal is the wrong floor for ARD, where a short length scale
along one coordinate is the signal rather than the pathology. Any caller enabling ARD must revisit
its HPO box; see 6.4.

### 6.4 Checklist before enabling ARD in a downstream campaign

Adapted from what `RT_ML4F` would need. Each item is a real coupling, not a formality.

1. **The HPO box.** A scalar `l_c` floor exists to prevent a diagonal kernel. Under ARD it also
   prevents a coordinate from being reported as relevant. Decide the floor per coordinate, or widen
   it, and state the reasoning. (`gp_theta_bounds_log` in RT's `contracts.py` broadcasts a 3-tuple,
   so this is a value decision, not a code change.)
2. **Any radius derived from `l_c`.** `h = kappa * l_c / sqrt(d/3)` becomes `h_i ∝ l_i` — a sphere
   in the rescaled metric is an ellipsoid in the original coordinates. A scalar radius under a 147×
   spread is far too tight in the inert coordinates and far too loose in the active one.
3. **Any code that formats or stores `l_c`.** `f"{gp.l_c:.3e}"` raises on an array. Use
   `bo.format_length_scales`. Diagnostics that report a floor-hit fraction should report it per
   coordinate.
4. **Any code that assumes `theta_log` has 3 entries.** RT's `PolicySearchConfig.validate` asserts
   `len(gp_theta_bounds_log) == 3`; that assertion stays valid thanks to broadcasting, but a
   `theta_log` read back from a checkpoint will have `d + 2` entries.
5. **Resume / checkpoint metadata.** A campaign resumed with a different `ard` setting must be
   rejected by the config-metadata comparison, not silently reshaped.
6. **Escape channel.** If ARD is used to down-weight coordinates, keep a channel through which the
   data can contradict it. Down-weighting is recoverable; dropping is not. This is why rescaling is
   preferable to projection.
7. **Freeze or declare.** If length scales are re-estimated during a campaign, do it at
   pre-declared checkpoints applied identically to every arm, and archive the realised values per
   round.

---

## 7. Reproducibility

`optimize_gp_hyperparams` drew its random restarts from an unseeded
`np.random.default_rng()`. No campaign using HPO was bit-reproducible, in any arm, with or without
ARD. `GPConfig.hpo_random_state` fixes this without changing the default: `None` reproduces the
historical behaviour exactly, an integer makes the restarts — and therefore the whole optimisation —
deterministic.

This is independent of ARD and is arguably the more consequential change in this patch. Downstream
studies that need reproducible campaigns should set it whether or not they enable ARD.

---

## 8. Tests

`tests/test_ard.py`, 52 tests, ~17 s (plus the 5 pre-existing tests in `tests/test_batch_ei_adam.py`). Grouped by what each group protects:

| group | what it guarantees |
|---|---|
| 1. Backwards compatibility | scalar `l_c` reproduces frozen copies of the pre-ARD kernels **exactly**; default config stays a float; `theta_log` and `param_hist` layouts unchanged; the pre-existing batch-EI/ADAM regression file still passes |
| 2. Kernel algebra | ARD with equal length scales == isotropic; `k_ARD(x,x';l) == k_iso(x/l,x'/l;1)`; diagonal is `sigma_f^2`; PSD; a short length scale decorrelates that coordinate and only that one; `d = 1` degenerates correctly |
| 3. Validation | non-positive / non-finite / wrong-length length scales rejected; HPO-box broadcasting and its rejection cases; whitening weights have unit geometric mean and rank coordinates correctly |
| 4. Marginal likelihood | layout inferred from `theta_log` length; NLL matches an independent Cholesky computation to 1e-12; ARD beats isotropic by >5 nats on anisotropic data |
| 5. HPO | recovers the relevant coordinates on synthetic anisotropic data; stays inside the declared box; `hpo_random_state` makes the fit reproducible; unseeded still works; a stale warm start is discarded not reshaped; scalar seeds expand |
| 6. Prediction | ARD held-out RMSE < 0.7 × isotropic on anisotropic data; frozen ARD == fitted ARD == whitened isotropic, to 1e-9 on mean and variance |
| 7. Rank-1 update | the `O(n^2)` extension reproduces the `O(n^3)` refit under a vector length scale; the vector survives the extension |
| 8. BO driver | end-to-end run with `ard=True`; history records `d` length scales; isotropic history stays scalar; logging and `.npz` export survive a vector |
| 9. Offline estimator | return contract, shapes, and malformed-input rejection |

Run with `pytest tests/ -q` from the repository root.

Two tests deserve comment. `test_scalar_length_scale_is_bit_identical_to_legacy` embeds the old
kernel formulas verbatim; if a future refactor unifies the isotropic and ARD branches, that test
fails and the decision becomes explicit rather than accidental.
`test_rank1_update_matches_full_refit_under_ard` runs with `normalize_y=False` on purpose: the
rank-1 path deliberately keeps the normalisation statistics it was built with, so with normalisation
on, the two paths are different models and the comparison would be meaningless.

---

## 9. What this does not establish

* **Nothing here is evidence about any particular objective.** The tests use synthetic anisotropic
  functions. The RSPA numbers quoted in section 1.2 were produced with scikit-learn, not with this
  library; the structure transfers, the numbers do not. Reproducing them here is now possible and is
  the obvious next step.
* **ARD does not make a non-stationary kernel stationary.** If a coordinate enters through a
  resonance band, a short fitted length scale may be standing in for a feature the kernel cannot
  represent at any single scale. ARD improves such a fit without making it correct.
* **A length scale at a bound is not a measurement.**
* **No claim is made about ARD helping in-campaign.** Section 6.3 measures the opposite at small `n`.
* **The acquisition optimiser is unchanged.** Candidate generation is still isotropic in the original
  box; nothing in this patch makes proposals respect the anisotropic metric. Callers that want that
  must rescale their own candidate sampling.

---

## 10. Repository reorganisation, same session

The library files stayed at the repository root, because `RT_ML4F/_external_paths.py` puts the repo
root on `sys.path` and imports `BO_ML4F` / `BO_func_YL` by name. Everything else moved:

| was | is |
|---|---|
| `BO_ML4F_documentation.{tex,pdf}` | `docs/` |
| `BO_ML4F_documentation.{aux,log,out,toc,synctex.gz}` | `docs/_build/`, untracked |
| `1D_Test_CASE.py`, `3D_Test_CASE.py`, `make_animation_1D.py`, `BO_function_tutorial.png` | `examples/` |
| `test_BO_ML4F_hybrid.py` | `tests/test_batch_ei_adam.py` |
| `BO_HOML26/`, `Single_Fid_BO_last_version/` | `legacy/` |
| `out/`, `out_3d/` | `_generated/`, untracked |

Added: `.gitignore`, `.gitattributes`, `conftest.py` (puts the repo root on `sys.path` for tests),
and a `sys.path` bootstrap in the two example scripts so they still run from `examples/`. Both
examples were re-run after the move and produce the same output as before.

**Line endings: nothing to do.** An early reading in this session suggested the repository had a
`CRLF` / `LF` problem, because plain `git diff` reported all 47 tracked files as fully rewritten
(9305 insertions, 9305 deletions). That was a stale **index**, not the repository: `git diff`
compares the working tree against the index, and this repository's index had been written by a
Windows git with `CRLF` endings while both the working tree and the committed history were `LF`.
`git diff HEAD` — working tree against history — was clean the whole time.

Verified after the reorganisation: **0 of 54 tracked text files contain `CRLF` in git history**, and
the working tree matches. The `git mv` / `git rm --cached` operations above rewrote the index
entries and the phantom diff is gone. `git add --renormalize .` is **not** needed and should not be
run.

The `.gitattributes` added in this session (`* text=auto eol=lf`) is kept as a guard, so that a
Windows editor cannot reintroduce the mismatch in future files. It is not a fix for anything
currently broken.

What `git diff --stat` reports now is the real work: 4 files changed, 482 insertions, 48 deletions —
`BO_ML4F.py`, `README.md`, and the two example scripts. The larger deletion count visible in
`git diff HEAD` is the generated output being untracked (`out/run.log`, `out_3d/run.log`, the LaTeX
build products); those files remain on disk.

Nothing in this session has been committed.
