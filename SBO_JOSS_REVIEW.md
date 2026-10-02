# pyRAMBO.sbo — deep review and JOSS-readiness plan

> **Audience:** a Claude Code session (or a human) that will *act* on this review.
> **Nothing in the repository was changed to produce it** — this file is the only addition. All probe scripts ran from a scratch folder outside the repo.

| | |
|---|---|
| Reviewed | `src/pyRAMBO/sbo/*` (core, refinement, persistence, plotting, saving), its tests, `simple_cases/sbo`, `advanced_cases/burgers_control`, packaging, README, `paper/` |
| Not reviewed in depth | `mfbo`, `mobo` internals (only grep-checked for "does the same defect exist there?"), the Burgers environment code, the LaTeX presentation |
| Date / state | 2026-10-01 · branch `yannick-latest-version` · HEAD `7d029d5` **plus a large uncommitted working tree** (see SBO-H01) |
| Environment | Windows 11 · Python 3.12.12 · NumPy 2.4.1 · SciPy 1.16.3 · Matplotlib 3.10.8 · pytest 9.1.1 · scikit-optimize 0.10.2 · scikit-learn 1.7.1 (only Windows was exercised) |
| Method | read every sbo source file; ran the test-suite (98 pass); ~30 targeted probes; validated the GP / acquisitions against scikit-learn and Monte-Carlo; sanity benchmark vs scikit-optimize; coverage / flake8 / mypy; wheel build + clean install; JOSS requirements fetched from the JOSS docs on 2026-10-01 |

---

## 0. How to use this file

1. Read §1–§4 (verdict, index, quick wins, JOSS scorecard). Then work item by item; every item has a stable ID.
2. **IDs:** `SBO-B` bug · `R` robustness · `P` performance · `D` design/API · `T` tests/CI · `E` examples/docs · `J` JOSS/paper · `H` repo hygiene.
3. **Severity:** **P0** = silently wrong results, or a hard JOSS blocker · **P1** = fix before submitting · **P2** = improvement. **Effort:** S < 1 h · M ≈ half a day · L = 1–3 days · XL > 3 days.
4. **Evidence tags:** **RAN** = reproduced by executing code · **READ** = from reading code only (re-verify before fixing) · **EXT** = from external web pages fetched 2026-10-01.
5. Tick the boxes in the index (§2) as items are closed and append a line to the *Progress log* (end of file).
6. **Repo conventions you must respect** (from `SBO_CHANGELOG_FOR_PORT.md`): `sbo` is the *reference* implementation; after every major sbo change append a numbered entry there (**next free number: 16**) and mark port status for `mobo`/`mfbo`. Items 12–15 of that file are still unported.
7. **Ground rules while fixing:** write the failing regression test first, then fix; keep `pytest` green (98 tests at review time); preserve the names in `sbo.__all__` and the on-disk formats (`trace.npz` columns, `load_trace` backward-compat, snapshots) unless an item says otherwise; **`BOResult.states[it].gp` must stay "the GP the acquisition used at iteration `it`"** (tests and `main_persistence_demo.py` rely on it); do **not** commit, push, `git stash`, `git checkout`, `git reset` or `git clean` — the working tree holds weeks of uncommitted work; ask the user first (SBO-H01).
8. Gotchas: running an example writes into `simple_cases/_output` (not git-ignored yet, 39 MB already there); `tests/test_plotting.py` needs a LaTeX install until SBO-B07 is fixed; environment/commands are in Appendix D.

---

## 1. Verdict

**The numerical core is correct; the package is not JOSS-ready yet.** It is a clean, compact, NumPy/SciPy-only GP-BO implementation whose maths I could validate against independent references (§A). What stands between it and a JOSS submission is mostly *not* the algorithm:

1. **The 6-month public-history rule is the critical path.** JOSS wants the repository public for more than six months with active development (EXT). Anonymous access to `github.com/mendezVKI/BayesianOptimization_ML4F` fails (page 404, API 404, `git ls-remote` asks for credentials) — it appears **private**. The clock has not started. → SBO-J01.
2. **`paper/paper.md` is far from the current JOSS format**: ~374 words (guidance 750–1750), TODO placeholders (ORCIDs, affiliation, statement of need, acknowledgements), no in-text citations, and three now-required sections are absent (*State of the field*, *Research impact statement*, *AI usage disclosure*). → SBO-J02…J05.
3. **A reviewer's first 15 minutes would go badly:** 7 of the 8 example scripts fail to import after the half-finished `examples/`→`simple_cases/` rename (SBO-E01); 5 of 6 plotting tests fail on a machine without LaTeX because `plotting.py` forces `usetex=True` at import (SBO-B07); there is no CI, no docs site, no CONTRIBUTING / CODE_OF_CONDUCT / CITATION.cff / CHANGELOG / release tag.
4. **Real defects exist** — most importantly `AcqConfig(maximize=True)` returns the *worst* point as "best" (SBO-B01), the objective-call accounting is wrong whenever gradient refinement merges a pair (SBO-B02), and the UCB batch heuristic collapses to duplicates (SBO-B03).
5. **The paper's headline claims need evidence the repo does not yet contain**: "rank-one acceleration" is *inactive in the default configuration* and not the bottleneck when active (SBO-P01); "parallel batch evaluation" is not supported by the driver (SBO-D02); gradient refinement is not demonstrated in any user-facing example and has no cost-matched baseline (SBO-P06).
6. **Feature gap vs the state of the field**: only an isotropic RBF kernel (no Matérn, no ARD). A Matérn-5/2 kernel and multi-restart HPO existed in the earlier monolith (`git show 85026d5:BO_ML4F.py`) and were lost in the port (SBO-D01).
7. **All of the Sept–Oct work is uncommitted** (SBO-H01).

Rough effort for the engineering + paper work: **6–10 focused weeks**, well inside the 6-month waiting period — so *make the repo public early* (after hygiene, SBO-H02/J09) and use the wait productively.

---

## 2. Master index

| ☐ | ID | Title | Sev | Eff | Evidence | Also in mobo/mfbo? |
|---|---|---|---|---|---|---|
| ☐ | SBO-B01 | `maximize=True` half-implemented: driver reports/uses the *minimum* | **P0** | S | RAN | mobo has a mirror test; mfbo unchecked |
| ☐ | SBO-B02 | Objective-call accounting wrong with refinement merges; gradient calls uncounted | P1 | M | RAN | n/a |
| ☐ | SBO-B03 | Batch acquisition: UCB collapses to duplicates; EI batches barely diverse | P1 | M | RAN | n/a |
| ☐ | SBO-B04 | `BOState.acq` aliases the live GP; `res.gp` is stale by the last iteration | P1 | M | RAN | likely (deepcopy pattern) |
| ☐ | SBO-B05 | Second-resolution run-folder names collide → `FileExistsError` | P1 | S | RAN | yes (grep) |
| ☐ | SBO-B06 | Log file handle never closed; global logger `"BO"` | P1 | S | RAN | yes (grep) |
| ☐ | SBO-B07 | Import-time `usetex=True`; plotting imported even when disabled; tests need LaTeX | **P0** | M | RAN | yes (grep) |
| ☐ | SBO-B08 | 1D state plot crashes without `f_true` (after the expensive initial design) | P1 | S | RAN | unchecked |
| ☐ | SBO-B09 | `plt_all` mutates the caller's `SaveConfig`; plots silently discarded | P2 | S | RAN | unchecked |
| ☐ | SBO-B10 | Convergence/exploration plot semantics ("Calls n" is the iteration index, etc.) | P2 | S | READ | unchecked |
| ☐ | SBO-B11 | Vector-valued objective silently truncated; duplicated scalar-conversion logic | P2 | S | RAN | unchecked |
| ☐ | SBO-R01 | No bounds validation (reversed bounds ⇒ GP carries no information ⇒ random search) | P1 | S | RAN | unchecked |
| ☐ | SBO-R02 | Config errors surface only *after* the initial design is evaluated | P1 | S | RAN | unchecked |
| ☐ | SBO-R03 | No handling of NaN/failed objective values (cryptic Cholesky error) | P1 | M | RAN | unchecked |
| ☐ | SBO-R04 | Crash safety / resume / reproducibility metadata (nothing saved if `f` raises) | P1 | L | RAN | partly |
| ☐ | SBO-R05 | Cholesky failure unhandled outside HPO; rank-one silently clips | P2 | S | READ | unchecked |
| ☐ | SBO-R06 | Conditioning, duplicate sampling and no stopping rule on noise-free objectives | P2 | M | RAN | unchecked |
| ☐ | SBO-R07 | Unguarded memory blow-up (grid candidates) | P2 | S | RAN | unchecked |
| ☐ | SBO-P01 | "Rank-one" is inactive by default and not the bottleneck | P1 | M | RAN | n/a |
| ☐ | SBO-P02 | `BOResult.states` deep-copies the dense GP every iteration (O(n_iter·n²) memory) | P1 | M | RAN | yes (grep) |
| ☐ | SBO-P03 | Kernel via 3-D broadcast and variance via `cho_solve` are 2–3× slower than necessary | P2 | S | RAN | unchecked |
| ☐ | SBO-P04 | Default acquisition optimizer (`random`) is weak; `refined` is slow | P1 | M–L | RAN | partly |
| ☐ | SBO-P05 | HPO: single start, finite-difference cost, no priors | P2 | M | RAN | mobo has restarts? unchecked |
| ☐ | SBO-P06 | Gradient refinement: no stopping rule, wasted evaluations, no cost-matched baseline | P2 | M | RAN | n/a |
| ☐ | SBO-D01 | Only isotropic RBF; Matérn-5/2 + multi-restart HPO existed before the port | P1 | L | READ+git | neither has Matérn |
| ☐ | SBO-D02 | No ask/tell interface; batch "parallelism" not realised by the driver | P1 | L | READ | unchecked |
| ☐ | SBO-D03 | Budget semantics (`n_iter` ≠ evaluations) and stopping criteria | P1 | M | READ | unchecked |
| ☐ | SBO-D04 | API ergonomics and documentation of the public surface | P1 | M | RAN/READ | partly |
| ☐ | SBO-D05 | Dead / misleading public names | P2 | S | READ | partly |
| ☐ | SBO-D06 | State the scope/limitations explicitly | P2 | S | — | — |
| ☐ | SBO-T01 | No CI | **P0** | M | RAN | — |
| ☐ | SBO-T02 | Coverage holes (75 % branch; PI/UCB, grid/refined, `init_dataset`, 2D plot, maximize…) | P1 | L | RAN | — |
| ☐ | SBO-T03 | Benchmark / validation suite and regression thresholds | P1 | M | RAN | — |
| ☐ | SBO-T04 | Static quality: mypy 17, flake8 5, docstring coverage 48 % | P2 | S–M | RAN | — |
| ☐ | SBO-E01 | `examples/`→`simple_cases/` rename half-done; 7/8 examples fail | **P0** | S | RAN | yes |
| ☐ | SBO-E02 | Example content issues (invalid options, mislabels, non-reproducible benchmark) | P1 | M | RAN | partly |
| ☐ | SBO-E03 | No docs site / statement of need / API reference | P1 | L | — | — |
| ☐ | SBO-E04 | Burgers case: third-party code, undeclared deps, no gradient demo | P1 | M | RAN | — |
| ☐ | SBO-J01 | Public-history requirement (repo appears private) | **P0** | decision | EXT+RAN | — |
| ☐ | SBO-J02 | `paper.md` structure/length/front-matter/citations | **P0** | L | RAN+EXT | — |
| ☐ | SBO-J03 | *State of the field* section | **P0** | L | EXT | — |
| ☐ | SBO-J04 | *Research impact statement* | **P0** | M | EXT | — |
| ☐ | SBO-J05 | *AI usage disclosure* | **P0** | S | EXT | — |
| ☐ | SBO-J06 | Releases, tags, changelog, Zenodo, PyPI, CITATION.cff | P1 | M | RAN | — |
| ☐ | SBO-J07 | Community files (CONTRIBUTING, CODE_OF_CONDUCT, templates) | P1 | S | RAN | — |
| ☐ | SBO-J08 | Claims-vs-evidence audit of README/paper statements | P1 | M | RAN | — |
| ☐ | SBO-J09 | Licence / ownership / third-party material | P2 | S | READ | — |
| ☐ | SBO-J10 | Scope & significance positioning | P1 | — | — | — |
| ☐ | SBO-H01 | Weeks of work are uncommitted | **P0** | S | RAN | — |
| ☐ | SBO-H02 | Stale / heavy material; repo inside OneDrive; repo vs package name | P1 | S | RAN | — |
| ☐ | SBO-H03 | Polish: typos, boilerplate headers, `print` in library, version in 3 places | P2 | S | RAN | — |

---

## 3. Quick wins (≈ 1 day, no design decisions needed)

1. **SBO-E01** — finish the rename (README, `simple_cases/README.md`, `paper/paper.md`, `.gitignore`, 7 scripts, docstrings) and add an import-smoke test for every example.
2. **SBO-B07** — remove module-level `plt.rc(...)`, import plotting lazily, auto-detect LaTeX.
3. **SBO-B01** — one `_best_index(y, maximize)` helper at three call sites + a mirror test.
4. **SBO-B05 / B06** — unique run folders; close the log handler (`try/finally`).
5. **SBO-R01 / R02** — `validate_bounds()` / `validate_configs()` before any objective call.
6. **SBO-E02** — replace the invalid `global_method="refine"`, fix `sinusoidal_1d`'s ignored `rng`, relabel the "With Rank-One Update" prints.
7. **SBO-H01** — ask the user how to commit the work (logical commits), then push to the branch.
8. **SBO-J06/J07** — add `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, and a minimal GitHub Actions workflow (SBO-T01).

---

## 4. JOSS scorecard (criteria from the JOSS docs, fetched 2026-10-01 — paraphrased; re-check before submitting)

| Criterion | Status | Notes / items |
|---|---|---|
| OSI-approved licence file | ✅ | MIT, `LICENSE` ships in the wheel. Copyright holders are the two individuals — see SBO-J09 |
| Public repo, open issue tracker, cloneable without registration | ❌ / unknown | appears private — SBO-J01 |
| > 6 months public history, active development, not a single burst | ❌ not demonstrable | 28 commits 2026-01-28 → 2026-08-27 (+ uncommitted Sep–Oct work); public date unknown — SBO-J01, H01 |
| Research software, substantial scholarly effort, "feature-complete" | ⚠️ | sbo ≈ 2.8 k lines (all three packages ≈ 7.2 k); feature gaps (kernel, ask/tell, docs) — SBO-D01/D02/E03/J10 |
| Pip-installable, dependencies documented | ⚠️ | wheel builds and installs (RAN); no PyPI release/tag; deps unpinned; LaTeX trap — SBO-J06, B07 |
| Reviewers can install and verify core functionality | ⚠️ | quickstart works; examples broken; plotting needs LaTeX — SBO-E01, B07 |
| Automated tests (ideally in CI) | ⚠️ | 98 tests pass locally; **no CI**; fail without LaTeX; 75 % coverage — SBO-T01/T02 |
| Statement of need in the docs | ❌ | README has none — SBO-E03 |
| API documentation | ❌ | 48 % of public callables have a docstring, no NumPy-style sections, no site — SBO-D04, E03 |
| Example usage | ❌ | 7/8 broken; headline features not demonstrated — SBO-E01/E02 |
| Community guidelines | ❌ | none — SBO-J07 |
| Paper: Summary / Statement of need / State of the field / Software design / Research impact / AI disclosure / references | ❌ | 3 of 6 sections present (Statement of need is a TODO) — SBO-J02…J05 |
| Releases / changelog | ❌ | no tags, no CHANGELOG — SBO-J06 |

---

## 5. Findings

### 5.1 Bugs

#### SBO-B01 — `AcqConfig(maximize=True)` is only half-implemented · P0 · S · RAN
- **Where:** `src/pyRAMBO/sbo/core.py::bayesian_optimization` — three `np.argmin(y)` sites: loop top (≈L1116, feeds `make_acquisition`), post-update best (≈L1165, feeds `BOState`, trace, history, plots), final result (≈L1288). `make_acquisition` itself *is* correct for `maximize` (verified: EI(min f) ≡ EI(max −f) exactly).
- **Problem:** with `maximize=True` the driver still takes the minimum as incumbent: EI/PI measure improvement against the *worst* observation, and `res.best_x/best_y`, `res.trace.y_best`, `history[*]["best_y"]`, `BOState.y_best` and `plt_conv` all report the minimum.
- **Evidence:** `f(x) = −(x−0.7)²`, bounds `[−2,2]`, `n_init=5, n_iter=12, seed 0`: `res.best_y = −6.14 == y.min()` while `y.max() ≈ −0.0000`; `res.best_x = −1.78` (true argmax 0.7). Minimising `−f` instead gives `best_y = 1e-4` at `x = 0.691`.
- **Fix:** helper `_best_index(y, maximize)` (argmax if maximize else argmin) used at the three sites; keep `res.y` in the user's sign. Don't negate `y` globally (plots/trace expect the user's sign).
- **Test:**
  ```python
  def _kw():
      return dict(bounds=[(-2.0, 2.0)], bo_cfg=bo.BOConfig(n_init=5, n_iter=10, random_state=0),
                  gp_cfg=bo.GPConfig(), optim_cfg=bo.OptimConfig(n_raw_samples=300), save_cfg=bo.SaveConfig())

  def test_maximize_mirrors_minimize_of_negated_objective():
      f = lambda x: float(-(x[0] - 0.7) ** 2)
      r_max = bo.bayesian_optimization(f=f, acq_cfg=bo.AcqConfig(maximize=True), **_kw())
      r_min = bo.bayesian_optimization(f=lambda x: -f(x), acq_cfg=bo.AcqConfig(), **_kw())
      np.testing.assert_allclose(r_max.y, -r_min.y, atol=1e-8)       # same search, mirrored (expected: NLL and EI are sign-symmetric)
      assert r_max.best_y == pytest.approx(r_max.y.max())
      assert np.all(np.diff(r_max.trace.y_best) >= -1e-12)
  ```
  (`tests/test_mobo_core.py::test_maximization_flag_mirrors_the_run` is the analogue.)

#### SBO-B02 — Evaluation accounting is wrong when refinement merges pairs; gradient calls are never counted · P1 · M · RAN
- **Where:** `refinement.py::refine_candidates`, `handle_close_pair`; driver: `y_proposed` is evaluated *before* refinement (≈L1129), `log_summary(n_evals=len(y))` (≈L1293), `history["n"]`.
- **Problem:** every proposal is evaluated first. If the (proposal, refined) pair is "close" it is merged into **one stored row**, but 2 (policy `"final"`) or 3 (policy `"midpoint"`) objective calls were spent. `len(res.y)`, the log line "Finished: N evaluations" and `history["n"]` under-report. The `n_steps` gradient calls per candidate per iteration (adjoint solves in the target use-case) are not counted anywhere. For expensive simulators budget accounting is the whole point, and any benchmark in the paper would be unfair without it.
- **Evidence** (counting wrappers; `n_init=4, n_iter=3, n_candidates=2, n_steps=20`):

  | policy / threshold | `f` calls | `len(res.y)` | gradient calls |
  |---|---|---|---|
  | `final`, forced merge | 16 | 10 | 120 |
  | `midpoint`, forced merge | 22 | 10 | 120 |
  | never merge | 16 | 16 | 120 |
- **Fix:** (1) wrap `f` and `gradient` in counting closures inside the driver; expose `BOResult.n_f_evals`, `n_grad_evals`, plus cumulative per-iteration counts in `BOState`, `TraceLog`/`trace.csv` and the log summary. (2) **Lazy proposal evaluation**: refine first (gradient only), decide the merge, and evaluate `f(x0)` only if the pair is kept — under `"final"` a merged candidate then costs 1 call instead of 2 (2 instead of 3 for `"midpoint"`); this directly saves expensive simulations. (3) Use cumulative f-evaluations on the x-axis of `plt_conv` (see B10).
- **Test:** `res.n_f_evals == calls[0]` for all three cases above; with forced merge under `"final"` the loop makes exactly `n_candidates` calls per iteration.

#### SBO-B03 — Batch acquisition: UCB collapses; EI batches are barely diverse · P1 · M · RAN
- **Where:** `core.py::optimize_acquisition_batch` (`return values * penalty`).
- **Problem:** the multiplicative penalty `a·(1 − exp(−(d/s)²))` is only meaningful for `a ≥ 0` (EI, PI). For UCB/LCB, `a = −(μ − κσ)` is negative whenever objective values are large and positive; multiplying by a penalty → 0 *raises* the value at already-selected points, so the greedy loop re-selects them. For EI the penalty is too weak against EI's dynamic range (EI is extremely peaked).
- **Evidence** (1-D, 7 design points, `l_c=0.2`, 4 candidates, `batch_distance_scale=0.05`; min pairwise distance in the normalised box):

  | acquisition | y-offset 0 | y-offset 100 |
  |---|---|---|
  | EI | 0.014 | 0.014 |
  | PI | 0.019 | 0.019 |
  | UCB | 0.063 | **0.000** (x = −0.477, −0.477, −0.48, −0.477) |

  So EI/PI batches sit ~3× closer than the nominal scale and ~14× closer than the GP length-scale — the "diverse batch" headline feature is weak even where it works.
- **Partial fix, validated in a scratch run:** make the penalty shift-invariant, `f_pen = a_ref + (a − a_ref)·penalty`, with `a_ref` = min of the *first* random scan (consistent across the later local-search calls):
  ```python
  a_ref = None
  for _ in range(n_candidates):
      def penalized(Xcand):
          a = np.asarray(acq(Xcand), float).reshape(-1)
          if not selected:
              return a
          d = _min_normalised_distance_to(selected, Xcand)           # as today
          return a_ref + (a - a_ref) * (1.0 - np.exp(-(d / scale) ** 2))
      result = optimize_acquisition(penalized, bounds, optim_cfg, rng)
      if a_ref is None:
          a_ref = float(np.min(result.a[np.isfinite(result.a)]))
      ...
  ```
  UCB, offset 100: min distance 0.000 → 0.081 (scale 0.05) → 0.144 (scale = `l_c` = 0.2). **EI stays at 0.015** → needs a different mechanism. Options to evaluate (design decision for the authors): hard exclusion radius (mask candidates closer than `r_min` to selected points, also in the local search), local penalization (González et al. 2016), Monte-Carlo q-EI. My quick Kriging-believer prototype did **not** diversify EI either (min distance ≈ 0) — do not assume it works. Tie the default scale to the GP length-scale (normalised units) instead of a constant 0.05, and document the method as a heuristic.
- **Test:** sign-invariance (`picks(acq) == picks(acq + 100)`); minimum pairwise distance ≥ *c*·`batch_distance_scale` on a peaked-EI toy problem (choose *c* after the redesign).

#### SBO-B04 — Stale / aliased model objects handed to the user · P1 · M · RAN
- **(a) `BOState.acq` is a closure over the *live* `gp`** (`fit_gp` mutates it in place), whereas `BOState.gp` is a frozen `deepcopy`. After the run `states[i].acq` evaluates with the *final* GP. Evidence: `max|states[0].acq − acq(rebuilt from states[0].gp)| = 4.1e-2` — 100 % of that acquisition's maximum; `states[3]`: 3.0e-3; only the last state is consistent.
- **(b) `BOResult.gp` has not seen the last iteration's data:** it is fitted at the *start* of the last iteration (`res.X` has 11 rows, `res.gp.Xs` 10 for `n_init=5, n_iter=6`). Anyone using `res.gp` as "the final surrogate" gets a stale one.
- **Fix:** build `acq` from the frozen copy (`gp_state = copy.deepcopy(gp)`; `acq_state = make_acquisition(acq_cfg, gp_state, y_best_acq)`) — or, better, drop `gp`/`acq` from `BOState` and rebuild lazily from a `GPSnapshot` via `reconstruct_gp` (also solves SBO-P02). Refit the final GP on all data once at the end *without* HPO (reuse the last hyperparameters) and document it.
- **Test:** `res.gp.Xs.shape[0] == len(res.X)`; `states[k].acq(Xg) == make_acquisition(cfg, states[k].gp, states[k].y_best_acq)(Xg)` for all `k`.

#### SBO-B05 — Second-resolution run-folder names collide · P1 · S · RAN
- **Where:** `saving.py::setup_experiment_folder` (`strftime("%Y-%m-%d_%H-%M-%S")` then `os.makedirs(exp_path, exist_ok=False)`).
- **Problem:** two runs started within the same second with `create_timestamp=True` (the default) → `FileExistsError` on the second. Typical in multi-seed benchmark loops, tests, CI.
- **Evidence:** three back-to-back runs: run 0 OK; runs 1 and 2 `FileExistsError: [WinError 183]`.
- **Fix:** try `exp_path`, `exp_path_1`, `exp_path_2`, … (or add microseconds/PID).
- **Test:** three consecutive runs into the same `out_path` give three distinct `res.out_path`.

#### SBO-B06 — Log file handle never closed; one global logger · P1 · S · RAN
- **Where:** `saving.py::setup_logger` (`logger.handlers.clear()` without `close()`; `logging.getLogger("BO")` shared by every run and by all three subpackages); the driver has no `try/finally`.
- **Problem:** the `FileHandler` stays open after the run → on Windows the run folder cannot be deleted/moved (`PermissionError [WinError 32]`; also breaks pytest `tmp_path` cleanup and OneDrive sync); if the objective raises, no end marker or traceback reaches `log.log`; concurrent runs share one logger.
- **Evidence:** after a run, `shutil.rmtree(res.out_path)` → `PermissionError`; `logging.getLogger("BO").handlers == [FileHandler]`.
- **Fix:** close and remove existing handlers in `setup_logger`; wrap the run in `try/finally` that logs the exception (`logger.exception`) and closes the handler; use a per-run logger name (`f"pyRAMBO.sbo.{run_id}"`) or one library logger + a per-run handler removed at the end.
- **Test:** `shutil.rmtree(res.out_path)` succeeds after a run with `log_enabled=True`; after a raising objective the log's last line records the failure.

#### SBO-B07 — Importing the plotting module has global side effects; plotting is imported when disabled; tests need LaTeX · P0 · M · RAN
- **Where:** `plotting.py` top level: `plt.rc('text', usetex=True)`, `font.family='serif'`, tick/label sizes (≈L25–29); `core.py::bayesian_optimization` ≈L1013 imports `.plotting` **unconditionally** on every call; `setup_plotting_toggle`.
- **Problem:** (1) the first call of the driver — with every plot flag off — imports `pyplot` and permanently sets `rcParams["text.usetex"]=True` for the whole interpreter, so the *user's own later figures* then need LaTeX too; (2) any figure drawn by the library fails on a machine without LaTeX (reviewer laptops, CI, HPC); (3) the README and `paper.md` claim plotting is imported lazily "only when a run needs it" — false for sbo (and mobo/mfbo); (4) Matplotlib is a hard dependency although `core` does not need it.
- **Evidence:** fresh process: after `import pyRAMBO` pyplot is *not* imported; after one run with plotting OFF: `matplotlib.pyplot` imported, `text.usetex=True`, `font.family=['serif']`. Simulated PATH without LaTeX: `tests/test_plotting.py` → **5 of 6 tests fail** (`FileNotFoundError … 'cmr12.tfm'`).
- **Fix:** (a) no `plt.rc` at import; apply the style with `with plt.rc_context(style):` inside the plotting functions; add `SaveConfig.usetex: Optional[bool] = None` → auto-detect `latex` + `dvipng` via `shutil.which` (see `simple_cases/_common.latex_available`); (b) import `.plotting` only if any `plt_*_enabled` / `plt_all` is set; (c) either make Matplotlib an optional extra (`pyRAMBO[plot]`) or keep it required but never touch global rcParams; (d) add a no-LaTeX CI job.
- **Tests:** in a *subprocess* assert `"matplotlib" not in sys.modules` after a no-plot run; assert `rcParams` unchanged; produce all plots with `shutil.which` monkeypatched to return `None`.

#### SBO-B08 — The 1D state plot requires `f_true` · P1 · S · RAN
- **Where:** `plotting.py::plt_state_1D` (`axs[0].plot(Xplot[:, 0], y_true, …)` with `y_true=None`); the `f` argument is passed but unused.
- **Problem:** `SaveConfig(plt_state_enabled=True)` in 1-D without `f_true` → `ValueError: x, y, and format string must not be None` at iteration 0 — after the whole initial design (5 evaluations here, hundreds of CPU-hours in the target use-case) and the first iteration. Already noted in the changelog (item 9) but still open.
- **Fix:** skip the dashed curve when `f_true is None`; warn once at start instead of crashing mid-run.

#### SBO-B09 — `plt_all` mutates the caller's `SaveConfig`; plots computed and silently discarded · P2 · S · RAN
- `setup_experiment_folder` returns the **same object** when `out_path is None`, so `setup_plotting_toggle(save_cfg)` flips `plt_state/conv/hist/MLE` flags on the user's object — violating the documented "never mutated" contract. Evidence: flags `(F,F,F) → (T,T,T)`. With `out_path=None` and `show_plots=False` the plots are drawn (1.9 s for 2 iterations) and thrown away without a warning.
- **Fix:** `dataclasses.replace` in the `None` branch (or return a new cfg from `setup_plotting_toggle`); warn when plots are enabled but neither `out_path` nor `show_plots` is set.

#### SBO-B10 — Plot semantics · P2 · S · READ
`plt_conv` labels the x-axis "Calls $n$" but plots the iteration index (ignores `n_init`, batch size, refinement multiplicity — use `history["n"]` / cumulative f-evals, see B02); `plt_hist` plots `y_next` = first proposal only; `plt_state_2D` treats the first `bo_cfg.n_init` rows as "Initial dataset" (wrong when `X_init` has more rows); state plots hard-code `dpi=250` while conv/hist use `save_cfg.dpi`.

#### SBO-B11 — Objective output handling · P2 · S · RAN
`evaluate_objective` and two lambdas in the driver (≈L1130, ≈L1141) all do `float(np.asarray(val).reshape(-1)[0])`: a vector-valued objective is **silently truncated** to its first component (probe: `[(x−0.7)², 1e6]` accepted), and NaN/inf pass through (see R03). **Fix:** one shared `_scalar_objective(f)` wrapper that enforces `size == 1` and finiteness (policy in R03) and is used by the initial design, proposals and refinement.

### 5.2 Robustness

#### SBO-R01 — No bounds validation · P1 · S · RAN
- **Evidence:** `bounds=[(2.0, -2.0)]` (reversed) runs without error or warning: `NormalizationHelper` clips the denominator to `eps`, the GP sees normalised inputs up to **3.8e12**, its posterior sd equals the prior sd everywhere (1.979 at 7 grid points = `σ_f·y_std`) — the GP carries no information and BO silently degenerates to random search. `(1.0, 1.0)` works only by luck; a NaN bound gives a cryptic `ValueError: array must not contain infs or NaNs` from Cholesky.
- **Fix:** `validate_bounds()` — array-like of shape `(d, 2)`, finite, `lo < hi`, `d ≥ 1` — called at the top of the driver and in the public helpers; message names the offending dimension.

#### SBO-R02 — Config errors surface only after the expensive initial design · P1 · S · RAN
- `AcqConfig.kind`, `OptimConfig.method` / `global_method` / `optimizer`, `n_candidates` and `batch_distance_scale` are validated lazily: the `ValueError` arrives after `n_init` objective evaluations (probe: 6 evaluations spent). `init_sampling` and the refinement config *are* validated early (good). Some wrong values are accepted silently: `global_method="refine"` (used in 4 examples) is ignored when `method="random"`.
- **Fix:** `validate_configs(...)` before any evaluation; `Literal`/`Enum` types.
- **Test:** counting objective asserts **0 calls** before each `ValueError`.

#### SBO-R03 — Failed / non-finite objective values · P1 · M · RAN
- A NaN from the objective surfaces as `ValueError: array must not contain infs or NaNs` from `scipy.linalg.cholesky` deep inside `gp_fit` — no mention of the objective or of the offending `x`. Diverged simulations are routine in CFD campaigns.
- **Fix:** detect in the shared objective wrapper (B11); `BOConfig.on_failure = "raise" | "skip" | "penalize"` ("penalize" = worst observed + margin / constant-liar); record failures in the evaluation log (R04); never feed NaN to the GP; `raise` should name the point.

#### SBO-R04 — Crash safety, resume, reproducibility metadata · P1 · L · RAN
- **Evidence:** an objective that raises on its 8th call (run with `out_path` set) leaves **only** `log.log` (start marker only) and `res/meta.json` — all 7 evaluations are lost: the trace is written once at the end by default (`trace_flush_every=None`) and Tier-2 snapshots are opt-in and hold pre-update data. `trace.npz` stores only `x_next/y_next` of the *first* proposal per iteration; for batch/refinement runs the full dataset never reaches disk (probe: 23 evaluated points in memory; `res/` = `meta.json, trace.csv, trace.npz`). `meta.json` omits `refinement_cfg`, `X_init/y_init/top_up_to_n_init` and library versions, so a refinement run can't be reproduced from its metadata. The `states=` parameter is a **no-op** (only `.append`ed to) that looks like resume support.
- **Fix:** an append-only `res/evaluations.csv` (one row per *objective call*: iteration, source ∈ {init, proposal, refined, midpoint}, `x…`, `y`, wall time, failed flag), flushed after every evaluation; `resume_from=<run folder>` rebuilding `(X, y)` (and the RNG state if saved); write `refinement_cfg`, `pyRAMBO`/NumPy/SciPy versions (and git hash when available) to `meta.json`; default `trace_flush_every=1` when `out_path` is set (the trace is tiny); remove or implement `states=`. Naturally lands in SBO-D02.
- **Test:** a crash at call *k* leaves *k* rows on disk; resuming continues the run.

#### SBO-R05 — Cholesky failure outside HPO; silent clipping · P2 · S · READ
`gp_fit` calls `cholesky` once: a non-PD matrix (duplicates + tiny user-fixed `sigma_y`, `jitter=0`) raises `LinAlgError` and aborts. `_rank_one_update` silently clips the Schur complement to `1e-14` when it is ≤ 0. **Fix:** jitter-escalation loop (`1e-10 → 1e-4`, `warnings.warn`); when the rank-one Schur complement is non-positive fall back to a full refit with a warning. The library never calls `warnings.warn` anywhere (0 occurrences) — add warnings for these numerical fallbacks.

#### SBO-R06 — Noise-free objectives: conditioning, duplicate sampling, no stopping rule · P2 · M · RAN
- **Evidence** (deterministic objectives, default config, 70 iterations, 8 runs): HPO drives `sigma_y` to (or within 5× of) its lower bound (1.0e-4 normalised) in all 8 runs; `cond(K) ≈ 2e10–7e11`; BO re-samples the optimum at normalised distances down to **6e-5** (1-D) and 1.5e-3 (2-D). No failure was observed, but α is accurate to only ~5 digits and evaluations are wasted. There is no stopping criterion, so the full `n_iter` is always spent.
- **Fix:** see SBO-D03 (stopping rules); minimum-separation check of proposals against existing `X` (warn/stop/perturb); consider a higher `sigma_y` floor or relative jitter; document.

#### SBO-R07 — Unguarded memory blow-up · P2 · S · RAN
`_make_grid_candidates` has no size guard: `d=7`, `grid_n_per_dim=20` would allocate ≈ 72 GB (20⁷·7·8 B). **Fix:** raise a clear error beyond a cap. (Related temporaries: SBO-P03.)

### 5.3 Performance and scalability

#### SBO-P01 — "Rank-one" is inactive by default and is not the bottleneck · P1 · M · RAN
- **(a) Inactive:** `fit_gp` requires `not do_hpo`; with the defaults (`optimize_hyperparams=True`, `hpo_every=1`) the rank-one branch is **never** taken at any `n`. Probe: `_rank_one_update` called **0** times (even with `rank_one_threshold=0`); with `hpo_every=5` it was called 12 times in 15 iterations. It also needs `n > rank_one_threshold = 1000`.
- **(b) Not the bottleneck** (d = 5, 2000 random acquisition candidates):

  | n | full refit | rank-1 (1 point) | acquisition (2000 random) | acquisition (`refined`) | one HPO call |
  |---|---|---|---|---|---|
  | 250 | 3.4 ms | 0.2 ms | 29.6 ms | 140 ms | 0.38 s |
  | 500 | 15.8 ms | 0.9 ms | 61.4 ms | 238 ms | 5.15 s |
  | 1000 | 62.5 ms | 5.7 ms | 124.7 ms | 1455 ms | 8.61 s |
  | 2000 | 244 ms | 24.0 ms | 252 ms | 7130 ms | (not run) |

  So the update speeds up a step that is already 2–100× cheaper than the others: end-to-end gain is bounded by ≈ 1.5–2× with random acquisition and a few percent with `refined` acquisition or HPO every iteration. (It *is* ~10× faster than a full refit in isolation, and exact — `test_rank1_update_matches_full_refit`.)
- **(c) Naming:** the implemented update appends one row/column to the Cholesky factor (bordering), not a rank-one *modification*; fine, but explain it in the paper (e.g. Seeger 2004, Golub & Van Loan).
- **Actions:** (1) cut the larger costs first (P03, P04) — then the rank-one saving becomes relatively significant; (2) HPO schedule: HPO every iteration while `n ≲ 100`, then every *k* (5–10) — this is what makes rank-one usable in long runs; (3) add `benchmarks/rank_one_scaling.py` → time/iteration vs `n` for {full, rank-1} × {HPO every 1, every 10}; put the honest figure in README/paper; (4) revisit the "Rank-One Accelerated" wording unless the figure supports it (SBO-J08).
- **Test:** driver-level equivalence — same seed with `rank_one=True/False`, `hpo_every=100`, `rank_one_threshold=0` ⇒ identical `X, y` to ~1e-6.

#### SBO-P02 — `BOResult.states` stores a deep copy of the dense GP at every iteration · P1 · M · RAN + arithmetic
- **Where:** driver: `BOState(..., gp=copy.deepcopy(gp), ...)`, always on.
- **Problem:** memory ∝ `n_iter·n²` (each state keeps `L`, `alpha`, `Xs`, `Xs_norm`, …). Measured 9.3 MB for `n_init=30, n_iter=120` (d=2). Dense-`L`-only extrapolation: 0.57 GB (100 init / 500 it), 3.5 GB (100/1000), 8.7 GB (500/1000), **18.7 GB** (1000/1000) — exactly the large-`n` regime that motivates rank-one. It also contradicts the persistence module's own principle ("never store dense covariance/Cholesky factors"). Same pattern in mfbo/mobo (grep).
- **Fix:** make per-iteration full states opt-in (`SaveConfig.keep_states=False`) or store a `GPSnapshot` (O(n·d)) and rebuild `gp` lazily with `reconstruct_gp`. Keep the `states[it].gp` semantics (see §0.7).
- **Test:** `sum(array.nbytes)` over `res.states` grows linearly, not quadratically, with `n`.

#### SBO-P03 — Kernel and predictive variance are 2–3× slower than needed · P2 · S · RAN
`rbf_kernel` builds an `(n1, n2, d)` temporary: for `n=1000, N=2000, d=6` it takes **98.5 ms and 192 MB** peak vs **29 ms / 32 MB** with `scipy.spatial.distance.cdist(…, "sqeuclidean")` (max diff 1e-16). `gp_predict` computes the variance through `cho_solve` (two triangular solves): 34 ms vs **18 ms** with one `solve_triangular(L, Ks.T, lower=True)` and `var = k** − Σ w²` (max diff 1.3e-15). Both are drop-in; existing tests plus the reference tests of §A guard them.

#### SBO-P04 — The default acquisition optimizer is weak; the refined one is slow · P1 · M–L · RAN
- Default `OptimConfig(method="random", n_raw_samples=2000)` = best of 2000 random points, no local search. On Hartmann6 (6-D, 60 calls, 5 seeds) the median final regret is **0.187** (default) vs **0.027** (`method="refined"`) vs **0.009** (scikit-optimize default); Branin: 0.049 / 0.028 / 0.002 (Appendix B). The refined mode costs 1.45 s/iteration at `n=1000` and 7.1 s at `n=2000` (finite-difference gradients through the whole GP posterior).
- **Fix:** (1) make `"refined"` the default *(changes results/runtime — ask the authors)*; (2) analytic ∇EI/∇PI/∇UCB (closed form from ∇μ, ∇σ for the RBF kernel; O(n·d) extra) → ≈ (d+1)× fewer acquisition calls inside L-BFGS-B; (3) restarts from a mix of top-*k* **and** spread candidates (the top 10 of 2000 random points often cluster around one peak); (4) quasi-random (Sobol/LHS) raw samples.
- *Checked and fine:* `if res.success` in the local search does not discard improvements in practice — 0 of 250 local runs reported failure.

#### SBO-P05 — Hyperparameter optimisation · P2 · M · RAN
- Single warm-started L-BFGS-B with finite differences. On 40 random problems (d=1–4, n=8–40): median NLL gap to the best of 30 restarts 3e-10, 90th percentile 2e-7, **but 1/40 ended 2.85 nats above the best** (poor local optimum). Median **86** NLL evaluations per HPO call (max 236) ≈ 4 Cholesky factorisations per L-BFGS iteration — this dominates cost for `n ≥ 500` (8.6 s at `n=1000`). No priors on `(l_c, σ_f, σ_y)`: with `n_init=5` the MLE is weakly determined and often sits on a bound.
- **Fix:** optional restarts (`GPConfig.n_hpo_restarts`, the old monolith had `3`), adaptive `hpo_every`, looser `ftol`, optional weak log-normal priors (MAP).
- **Do not assume an analytic NLL gradient helps** for the 3-parameter isotropic kernel: it needs `K⁻¹` (≈ 3 Cholesky-equivalents vs 4 for finite differences). It pays off only with ARD (d + 2 parameters) — see SBO-D01.

#### SBO-P06 — Gradient refinement efficiency and evidence · P2 · M · RAN
- ADAM has **no stopping rule**: it always spends `n_steps` gradient calls (adjoint solves!) even at convergence → add `gtol`. Offer L-BFGS-B(`jac`) as an alternative local solver; accept `(value, grad)` callables and reuse the value (the code already tolerates tuples but discards the value). De-duplicate across candidates and against existing `X`, not only inside a (proposal, refined) pair — probe on a single-basin objective, 3 candidates, 8 iterations: 54 evaluated points, 30 pairs closer than 1e-2 (normalised). Related: lazy `y0` (SBO-B02).
- **Evidence for the paper:** on smooth benchmarks the ADAM-polished BO reaches regret ≈ 0.002 on Hartmann6 within ~20 calls — but it spent **750 gradient evaluations** (Appendix B). Without a baseline matched in total (function + gradient) cost — e.g. multi-start L-BFGS-B with the same gradient budget — reviewers will ask whether BO adds anything. Report both axes (f-calls and total cost) and state the gradient/function cost ratio assumed.

### 5.4 Design and API

#### SBO-D01 — Only isotropic RBF; Matérn-5/2 and multi-restart HPO existed before the port · P1 · L · READ + git
- Commit `85026d5` added `BO_ML4F.py` (`GPConfig.kernel = "rbf"/"matern52"`, `n_hpo_restarts=3`, `chol_rank1_update`, `alpha_rank1_update`, `update_gp_rank1`) and `BO_ML4F_documentation.tex` (implementation notes: "developed for the RSPA paper on the Burgers control case"; both deleted later in `6c3d433` / `2a85e06`). Inspect with `git show 85026d5:BO_ML4F.py` and `git show 85026d5:BO_ML4F_documentation.tex`. The current `sbo` has **one isotropic RBF** (a single length-scale for all dimensions, no ARD); `persistence._KERNEL_ID = "rbf_amp"` hard-codes it. Matérn-5/2 is the usual BO default (Snoek et al. 2012 — which the paper cites — recommend it) and ARD is standard when parameters have very different sensitivity.
- **Actions:** (1) establish which kernel/HPO settings produced the RSPA results and make sure the packaged `sbo` can reproduce them (tag that version); (2) restore Matérn-5/2 (+ optional ARD with analytic NLL gradient) via a small kernel registry — snapshots must then store kernel id + params; (3) pick the default after the benchmark in Appendix C.

#### SBO-D02 — No ask/tell interface; batch parallelism not realised · P1 · L · READ
Batch points are evaluated **serially** (`[f(x) for x in x_proposed]`); there is no hook for concurrent evaluation (executor / `map` / vectorised `f_batch`) and no way for an external scheduler (HPC queue, CFD farm) to take proposals and report results later. The paper's claim ("better exploit parallel evaluation budgets") only holds if the *user* parallelises inside `f` — which is called sequentially and therefore cannot overlap jobs.
**Fix:** a `BayesianOptimizer` class with `ask(n)` / `tell(X, y)` and serialisable state; `bayesian_optimization()` becomes a thin loop over it; optional `map_fn`/executor argument. This also yields resume (R04), failure policies (R03) and a natural home for counters (B02).

#### SBO-D03 — Budget semantics and stopping criteria · P1 · M · READ
`n_iter` counts *iterations*; objective calls = `n_init + n_iter·n_candidates·{1 or 2}` minus merges. Add `max_evals` / `time_budget` / `ei_tol` / `patience`; return and log counters (B02).

#### SBO-D04 — API ergonomics and public-surface documentation · P1 · M · RAN/READ
- `bayesian_optimization(f, bounds)` fails: `TypeError: missing 5 required positional arguments` — give every config a default (`None` → defaults) so a 3-line quickstart works.
- `OptimConfig.method` vs `global_method` is confusing (`"refined"` = random/grid scan + L-BFGS-B; `global_method="refine"` is accepted silently). Suggest `candidates ∈ {random, sobol, grid}` + `local_refine: bool`; `Literal`/`Enum` types.
- `AcqConfig.xi` is in objective units (unit-dependent) — document or scale by `y_std`.
- `BOResult`: expose all evaluated points with iteration/source labels; for noisy objectives `best_y` is the best *noisy observation* (the examples use noise 0.1) — add `best_x_posterior` / posterior-mean minimiser.
- Docs: only 34/71 public callables/classes have a docstring (48 %), none in NumPy-doc style; the config dataclasses — the main user surface — carry only `#` comments; `optimize_acquisition` is annotated `-> Array` and documented as returning `x_next` but returns `AcqOptimizationResult`.

#### SBO-D05 — Dead / misleading public names · P2 · S · READ
`SaveConfig.plt_MLE_conv_enbable` (typo; set by `plt_all`, never read, no such plot exists); `MLE_hist` / `param_hist` arguments of `negative_log_marginal_likelihood` / `optimize_gp_hyperparams` (never passed by the driver); the `states=` parameter (R04); `GPConfig.theta0_log` restarts *every* HPO from the same point instead of warm-starting (document); `BOState.x_next/y_next` and trace `x_next/y_next` are the first proposal only in batch mode; `adam_refine_candidate` re-exported from `core` "historically"; version string duplicated in `pyproject.toml`, `pyRAMBO/__init__.py` and `sbo/__init__.py`.

#### SBO-D06 — Scope statement · P2 · S
Box-constrained continuous inputs only: no constraints, integer/categorical variables, failed-evaluation model, ARD (yet). Say so plainly in README/paper — reviewers will ask about the constraints typical of CFD design problems.

### 5.5 Tests and CI

#### SBO-T01 — No CI · P0 · M · RAN
No `.github/`. Add GitHub Actions: matrix {ubuntu, windows, macos} × {3.10, 3.11, 3.12, 3.13}; a **min-deps** job (oldest supported NumPy/SciPy — `scipy.stats.qmc` needs SciPy ≥ 1.7, measure the real floor); a **no-LaTeX** job; `pytest --cov` with a threshold; ruff/flake8; wheel build + `twine check`; docs build. `requires-python = ">=3.9"` is declared but untested (source parses under 3.9 with `ast`; NumPy ≥ 2.1 / SciPy ≥ 1.14 need ≥ 3.10) — decide the floor.

#### SBO-T02 — Coverage holes · P1 · L · RAN
Branch coverage of `sbo` by the sbo-related tests: **75 %** (core 76 %, persistence 87 %, plotting **49 %**, refinement 89 %, saving 89 %). Not exercised by any test: PI and UCB acquisitions (core ≈L786–803), `optimize_acquisition` grid / refined paths (≈L852–921), `init_dataset` with `X_init`/`y_init`/top-up (≈L386–422), `gp_predict(return_cov=True)` (≈L483–490), flush-without-`out_path` errors, the entire 2-D state plot (plotting ≈L276–500), refinement-config validation branches, `maximize=True` (broken: B01), batch properties, reproducibility, failure paths. **Add:** a regression test per B/R item; the cheap reference tests of §A (GP vs scikit-learn, EI/PI vs Monte-Carlo); Hypothesis tests (K is PSD, EI ≥ 0, best-so-far monotone). Rename sbo test files to `test_sbo_*.py` like `test_mobo_*` / `test_mfbo_*` (today `test_core.py`, `test_persistence.py`, `test_plotting.py`, `test_refinement.py` are sbo's).

#### SBO-T03 — Validation / benchmark suite · P1 · M · RAN
Create `benchmarks/` (not collected by default pytest) implementing Appendix C with CSV/JSON output and a script that regenerates the paper figure; add a `slow`-marked regression test (e.g. Branin on its standard domain reaches regret < *X* in *N* calls for a fixed seed).

#### SBO-T04 — Static quality · P2 · S–M · RAN
mypy: 17 errors in sbo — meaningful ones: `optimize_acquisition` return annotation; `AcqOptimizationResult.a_best` typed `Array` but a float; `TraceLog.append(acq_value: float)` receives an `Array`; Optional narrowing on `GPModel` fields; (`np.savez(**dict)` hits are stub false-positives). flake8 (F/E9/C901): unused local `x_best_acq` (core ≈L1118), redefinition `local_gradient` (refinement ≈L179–184), one unused import (re-export — add `__all__`/`noqa`), `bayesian_optimization` is **322 lines, McCabe 17**, `plt_state_2D` 261 lines, `plt_state_1D` 131 — split the driver (setup / iteration / persistence / plotting hooks); 57 style hits at line length 120. Add ruff + pre-commit.

### 5.6 Examples and docs

#### SBO-E01 — `examples/` → `simple_cases/` rename is half-done; 7 of 8 examples fail · P0 · S · RAN
- **Evidence:** `ModuleNotFoundError: No module named 'examples.benchmarks'` for `simple_cases/sbo/{main_1D_sin, main_2D_branin, main_2D_rosenbrock, main_persistence_demo}.py`, `simple_cases/mfbo/main_mfbo_forrester.py`, `simple_cases/mobo/{main_mobo_binh_korn, main_mobo_schaffer}.py`. Only `sbo/main_1D_sin_large_scale.py` was updated (`from simple_cases import _common`). A stray empty `examples/_output/` (re-created by a run) remains.
- **Stale references:** `README.md` (links to `examples/`, `examples/README.md`, `examples/sbo/main_persistence_demo.py`, layout block, "examples only ever add the repository root … `examples.benchmarks`"), `simple_cases/README.md` (`python -m examples.mobo…`, layout), `paper/paper.md` ("tutorials in `examples/`"), `.gitignore` (`examples/_output/`, `examples/*_out/` …; `simple_cases/_output` — 39 MB — is **not** ignored), docstrings of `simple_cases/__init__.py`, `benchmarks.py`, `mfbo/__init__.py`, `mobo/__init__.py`, and the header comment of every script.
- **Fix:** choose one name (JOSS doesn't care; `examples/` is conventional), replace globally, and add a test that imports/runs every example with tiny budgets under `MPLBACKEND=Agg` so this cannot regress.

#### SBO-E02 — Example content issues · P1 · M · RAN/READ
- Invalid `global_method="refine"` in `main_1D_sin.py` (L110, L146), `main_1D_sin_large_scale.py` (L121), `main_2D_branin.py` (L112), `main_2D_rosenbrock.py` (L112); valid values are `"random"` / `"grid"`.
- `main_1D_sin.py` prints "Home-Made BO; With Rank-One Update", but with `optimize_hyperparams=True` and n ≈ 15 rank-one is never used (SBO-P01); a commented-out block uses a removed API (`plt_cfg=bo.PlotConfig()`); Spyder boilerplate headers; typos ("nois", "reproducability").
- `benchmarks.sinusoidal_1d` **ignores its `rng` argument** (global `np.random.randn`), so the seeded example is not reproducible (verified: same seed twice → different output; `sinusoidal_1d_large_scale` is fine).
- Branin is evaluated on `[−2, 2]²` instead of the standard domain `[−5,10]×[0,15]`, so the known optimum (0.3979) is not comparable — use the standard domain for the quantitative example.
- The headline features (batch acquisition, gradient refinement) are demonstrated only in `Presentations/make_figures.py` (Rosenbrock), not in a user-facing tutorial; no example for ask/tell, resume, maximisation or the best *posterior* estimate for noisy objectives; the README promises a "hyperparameter optimization" tutorial that does not exist.
- Every example calls `plt.show()`; `simple_cases/_common.use_paper_style` falls back to mathtext without LaTeX, but library plots still need LaTeX (SBO-B07).

#### SBO-E03 — Documentation site and statement of need · P1 · L
README has no statement of need, limitations or comparison; there is no API reference or theory page. Reuse the maths from `git show 85026d5:BO_ML4F_documentation.tex` as theory notes; Sphinx (autodoc + napoleon) or MkDocs-Material + mkdocstrings on Read the Docs; tutorials built from the examples and executed in CI; an "Installation" page that explains the optional LaTeX dependency.

#### SBO-E04 — Burgers advanced case · P1 · M · RAN
`advanced_cases/burgers_control/Burgers/` is third-party code ("Fabio and Lorenzo") with no licence/author header — confirm redistribution rights and attribute (acknowledgements or authorship). Undeclared dependencies: `gymnasium` (or `gym`), `scikit-optimize`, `imageio`; the legacy RL scripts import `tensorforce` / `stable_baselines` (obsolete; keep them out of the repo or isolate them). The study uses **no gradients**, so it does not showcase the refinement feature — add an adjoint/AD-gradient case or a differentiable surrogate. It *is* useful impact evidence: 10 runs × (10 init + 50 evaluations): sbo mean best cost **5282.7 ± 581.6** vs scikit-optimize **5191.6 ± 334.6** (indistinguishable); wall time 12.6 s vs 12.0 s (≈ 75 % spent in the objective). Keep generated `results/` (11 MB incl. GIFs) out of git or attach them to a release.

### 5.7 JOSS-specific

#### SBO-J01 — Public-history requirement · P0 · decision
- JOSS (EXT): the repository must have been public for more than six months before submission, with ongoing development over that period, openly cloneable, with a readable public issue tracker.
- **Evidence (RAN):** `https://github.com/mendezVKI/BayesianOptimization_ML4F` → 404; `https://api.github.com/repos/mendezVKI/BayesianOptimization_ML4F` → 404; anonymous `git ls-remote` → "Authentication failed". Consistent with a private repo (or a different URL). Git history: 28 commits (all branches), 2026-01-28 → 2026-08-27 (Yannick 16, Miguel 11, 1 unattributed); no tags; the Sept–Oct work is uncommitted.
- **Action (decision for the authors):** decide *when* to go public (after SBO-H02 and J09), then **tag the version that produced the RSPA results** and start cutting releases. Earliest realistic submission = public date + 6 months. Use the wait for external feedback (issues, a colleague's installation test), CI, docs and the benchmark study.

#### SBO-J02 — `paper/paper.md` · P0 · L · RAN + EXT
- **Today:** ~374 words in the body; placeholder ORCIDs (`0000-0000-0000-0000`) and a "TODO: confirm affiliation"; sections Summary, Statement of need (**TODO**), Software design, Acknowledgements (**TODO**), References; **no in-text citations** (the three bib entries are never cited); `date: 26 August 2026`; says "each subpackage has the same four modules" (sbo now has five — `refinement.py`) and "tutorials in `examples/`".
- **Required by JOSS now (EXT, paraphrased):** Summary (non-specialist audience); Statement of need; **State of the field** (comparison and build-vs-contribute justification); Software design (trade-offs, architecture, why it matters for the research); **Research impact statement** (realised impact or credible near-term significance); **AI usage disclosure**; key references including related software. Length guidance **750–1750 words**. Front-matter: title, tags, authors (`name`, `given-names`, `surname`, `orcid`, `affiliation`), affiliations (`index`, `name`, optional `ror`), `date` formatted like "1 October 2026", `bibliography`. Citations as `[@key]`.
- **Fix list:** fill ORCIDs/ROR/affiliation; write the four missing/placeholder sections (J03–J05); cite in the text; include **one benchmark figure** produced by `benchmarks/` (T03) and the rank-one scaling figure (P01); bring `paper.bib` to JOSS quality — `snoek2012practical` as `@inproceedings` with `booktitle`; `kingma2015adam` as `@inproceedings` (ICLR) or arXiv:1412.6980; add DOIs/URLs everywhere; add the references listed in J03.

#### SBO-J03 — State of the field · P0 · L · EXT (content from memory — verify every claim before publishing)
Compare honestly and justify "build vs contribute". Candidates: **BoTorch/Ax** (PyTorch; MC acquisition, q-batch, multi-fidelity, multi-objective, derivative GPs), **scikit-optimize** (maintenance status — verify), **GPyOpt** (archived), **Emukit**, **bayesian-optimization**, **SMAC3**, **Optuna**, **Dragonfly**, **HEBO**, **Trieste**, **SMT** (EGO, gradient-enhanced kriging), plus the literature on BO with gradients (Wu et al. 2017) and batch BO (González et al. 2016). Differentiators the authors can *defend*: (1) hybrid BO + user-supplied gradient (adjoint/AD) polishing of each proposal for expensive simulations; (2) NumPy/SciPy-only, readable implementation with an explicit bordered-Cholesky update; (3) two-tier persistence with exact posterior replay; (4) single-/multi-fidelity/multi-objective under one API. What does **not** differentiate (be candid): kernel choice, ARD, constraints, parallel infrastructure, acquisition sophistication — BoTorch is stronger on all of them. Explain why gradients are used *outside* the GP (polish) rather than inside it (derivative-enhanced GP).

#### SBO-J04 — Research impact statement · P0 · M · EXT
Available evidence to cite/collect: the RSPA paper on Burgers control (July-2026 revision; PR #1 `RSPA_paper_updates` in the history — **state its status and which code version it used**); `advanced_cases/burgers_control` (sbo ≈ scikit-optimize, 10 seeds); use inside the ML4F group (theses, projects); `Presentations/pyRAMBO_overview.pdf`. JOSS accepts "credible near-term significance" too, but concrete external use or a published application is much stronger. Add a `docs/` page listing users/applications and invite issues.

#### SBO-J05 — AI usage disclosure · P0 · S · EXT
Required section: which tools/models (with versions), the nature and scope of assistance (software, tests, documentation, paper), and an explicit statement that the human authors reviewed, edited and validated all AI-assisted output; authors stay fully responsible for correctness, originality and licensing. If Claude Code or any other generative-AI tool was used in this repository — this review was itself produced with Claude Code — the authors must disclose it truthfully. JOSS also expects the authors to demonstrate irreplaceable human contribution (problem framing, key design decisions, architecture), so the paper's *Software design* section should record the decisions the authors themselves took.

#### SBO-J06 — Releases, changelog, archive, citation · P1 · M · RAN
No git tags (`0.3.0` exists only in metadata); no `CHANGELOG.md` (the in-repo `SBO_CHANGELOG_FOR_PORT.md` is an internal porting checklist — move it under `docs/dev/`); no `CITATION.cff`; no Zenodo integration; no PyPI release — the name `pyrambo` returned 404 on `pypi.org` on 2026-10-01 (appears free; reserve it early). Version is duplicated in three places → single-source it (`importlib.metadata` or dynamic setuptools). `pyproject.toml` hygiene: `license = {text = "MIT"}` + License classifier (newer setuptools prefer the SPDX string form), no `keywords`, no Python-version classifiers, no `Documentation` / `Issues` URLs, no `plot`/`docs`/`dev` extras, no lower bounds on `numpy`/`scipy`/`matplotlib`; add `py.typed` if types are kept.

#### SBO-J07 — Community files · P1 · S · RAN
Add `CONTRIBUTING.md` (report issues, propose changes, run tests, style), `CODE_OF_CONDUCT.md`, issue/PR templates, a support statement. JOSS reviewers check for explicit community guidelines.

#### SBO-J08 — Claims-vs-evidence audit · P1 · M · RAN
| Statement (README / `paper.md`) | Status |
|---|---|
| "exact rank-1 Cholesky update scheme" | Exact ✔ (tested), **inactive by default and not the bottleneck** (SBO-P01) |
| "Rank-One Accelerated" (project name) | Not supported by the measurements above without HPO scheduling and cheaper acquisition |
| "dependency-light" | ✔ NumPy/SciPy/Matplotlib; but Matplotlib is mandatory and LaTeX is needed for library plots (SBO-B07) |
| "diverse batch acquisition … parallel evaluation budgets" | Heuristic with weak EI diversity, UCB broken, evaluation is serial (SBO-B03, D02) |
| "gradient-based local refinement … projected ADAM" | Implemented ✔; accounting and baseline missing (SBO-B02, P06) |
| "plotting.py imported lazily so headless/HPC use needs no backend" | **False** — imported on every run (SBO-B07) |
| "runnable tutorials in `examples/`" | Directory renamed; 7/8 scripts fail (SBO-E01) |
| "correctness checked in `tests/`" | ✔ 98 tests; coverage 75 % (T02) |
| "hyperparameter optimization … tutorials" | No such tutorial (E02) |

#### SBO-J09 — Licence, ownership, third-party material · P2 · S · READ
MIT with the two individuals as copyright holders — confirm with VKI that individuals may license the work (employer IP policy). Third-party code: `advanced_cases/…/Burgers/*` (E04), `legacy/*` (authorship?), benchmark-function implementations (cite Branin/Hartmann/Forrester sources).

#### SBO-J10 — Scope and significance positioning · P1
JOSS asks for substantial, feature-complete research software, not a minor utility. sbo ≈ 2.8 k lines (three packages ≈ 7.2 k), 28 commits. Frame the submission around what is distinctive (J03), close the credibility gaps (kernel, ask/tell, docs, CI, benchmarks), and decide whether the paper covers all three subpackages (broader, more to defend) or sbo plus brief mentions.

### 5.8 Repository hygiene

#### SBO-H01 — Weeks of work are uncommitted · P0 · S · RAN
`git status` (at review start): 1 added; 15 *renamed-and-modified* (staged `src/{sbo,mfbo,mobo}` → `src/pyRAMBO/…` with unstaged edits); 15 deleted (old `examples/`, unstaged); 9 modified; 7 untracked — `src/pyRAMBO/sbo/refinement.py`, `tests/test_refinement.py`, `tests/test_plotting.py`, `simple_cases/`, `advanced_cases/`, `Presentations/`, `SBO_CHANGELOG_FOR_PORT.md`. Last commit 2026-08-27; the whole Sept–Oct refactor exists only in the working tree (and OneDrive). **Ask the user** how to commit (suggested logical commits: package rename → refinement extraction → persistence/plot changes + tests → examples rename → advanced case), then push the branch.

#### SBO-H02 — Stale / heavy material; location; naming · P1 · S · RAN
`bo_ml4f.egg-info/`, `pyBO.egg-info/` (old package names; ignored), `legacy/` (2 scripts), empty `literature/`, `Presentations/` (PDF + tex + figures, 2.3 MB), `simple_cases/_output` (39 MB generated), `simple_cases/mfbo/mfbo_forrester_out`, `advanced_cases/…/results` (11 MB). The repo lives **inside OneDrive** (sync locks and conflicts — the code already works around a `PermissionError` in `_common.output_dir`): prefer a non-synced path. Repo name `BayesianOptimization_ML4F` ≠ package `pyRAMBO` — consider renaming the repo (GitHub redirects). Optional: "RAMBO" is also a well-known HEP phase-space generator — expect search-engine overlap.

#### SBO-H03 — Polish · P2 · S · RAN
Typos in comments/identifiers (`plt_MLE_conv_enbable`, "settigns", "expeted", "fromt eh", "convinient", "nois", "reproducability"); Spyder boilerplate headers; `print()` in library code (`plotting.py` ≈L103 → `warnings.warn`/logging); the library never warns (R05); version in 3 places (J06); consistent test-file naming (T02).

---

## 6. Roadmap

| Phase | Content | Items | Rough effort |
|---|---|---|---|
| 0 · Safety & hygiene | commit the work, finish the rename, hygiene, decide public date | H01, E01, H02, J09, J01 | 0.5–1 d |
| 1 · Correctness | the P0/P1 bugs, each with a regression test | B01–B08, B11, R01–R03 | 2–3 d |
| 2 · State & robustness | lightweight states, evaluation log/resume, counters, stopping rules, API defaults/docstrings | B02, B04, P02, R04–R07, D03–D05 | 3–4 d |
| 3 · Performance & defaults | cdist + triangular solve, acquisition defaults/gradients, HPO schedule/restarts, rank-one story | P01, P03–P05 | 2–4 d |
| 4 · Capability gaps | Matérn/ARD kernel registry, ask/tell + parallel map, refinement improvements | D01, D02, P06, B03 redesign | 1–2 wk |
| 5 · QA infra & docs | CI, coverage, benchmarks, docs site, community files, releases | T01–T04, E02–E04, J06, J07 | 1–2 wk |
| 6 · Paper & evidence | benchmark study, State of the field, Research impact, AI disclosure, claims audit | T03, J02–J05, J08, J10 | 2–3 wk |
| 7 · Port & submit | port infra fixes to mobo/mfbo (changelog items 16+), final re-check of JOSS docs, submit ≥ 6 months after going public | — | — |

**Definition of done ("JOSS-ready"):** repo public ≥ 6 months with tagged releases + CHANGELOG + Zenodo DOI; CI green on Linux/macOS/Windows × supported Pythons (+ min-deps, no-LaTeX); all P0/P1 items closed; examples run from a fresh clone; docs site with API reference and tutorials; `paper.md` has all required sections, 750–1750 words, in-text citations, ORCIDs, reproducible benchmark figure(s); AI disclosure; research-impact evidence; PyPI release.

## 7. Open questions for the authors (I could not decide these)

1. When can the repo go public — any IP, third-party-code (Burgers environment) or secrets review needed first?
2. Is the RSPA paper published/accepted? Which code version produced its results, and can it be tagged (e.g. `v0.2-rspa`)?
3. May the defaults change (acquisition `method`, `hpo_every`, kernel) and `OptimConfig` be renamed (breaking)?
4. Restore Matérn-5/2 + multi-restart HPO from `85026d5`? Which kernel should be the default?
5. Is the JOSS paper about all of pyRAMBO (sbo+mfbo+mobo) or mainly sbo?
6. Python/NumPy/SciPy support floor (3.9 is declared but untested)?
7. Keep `SBO_CHANGELOG_FOR_PORT.md` in the public repo (suggest `docs/dev/`)? Rename the repo to `pyRAMBO`? Move out of OneDrive?
8. Which batch-diversity mechanism do the authors want to stand behind (B03)?

---

## Appendix A — Verified correct (don't re-investigate; do keep these tests when refactoring)

| Check | Result |
|---|---|
| GP posterior mean/variance vs scikit-learn (`ConstantKernel·RBF + WhiteKernel`, fixed hypers), raw and with the normalising wrapper | max abs diff ≤ 2.3e-14 (raw: 1.8e-15 / 1.0e-15) |
| `gp_predict(return_cov=True)` vs scikit-learn | 5.8e-15 (mean), 1.1e-15 (cov) |
| Negative log-marginal-likelihood vs scikit-learn | diff 3.6e-15 |
| EI and PI vs Monte-Carlo (4·10⁶ draws) | within MC error (≤ 1.6e-5) |
| `make_acquisition`: EI(min f) ≡ EI(max −f) | exactly equal |
| Bordered-Cholesky update vs full refit (several new points) | `test_rank1_update_matches_full_refit` ✔; 11× faster in isolation at n=1000 |
| HPO objective optimum vs 30-restart reference | 39/40 within 2.3e-7 NLL; bounds prevent NaNs |
| Reproducibility with `random_state` | identical `y` for equal seeds; different for different seeds |
| SciPy/NumPy deprecation warnings (`LatinHypercube(seed=…)` etc.) | none on SciPy 1.16.3 / NumPy 2.4.1 |
| Persistence round-trip (live GP vs reconstructed posterior) | ≤ 1e-10; `np.load` without pickle; JSON metadata |
| Acquisition optimizers (`random`, `grid`, `refined`) | in-bounds points; `refined` ≥ `random` in acquisition value; 0/250 failed L-BFGS-B runs |
| Packaging | wheel builds from a clean copy (all modules + LICENSE), installs with `--target`, README quickstart runs from the installed wheel in another cwd (`best_y = 2.8e-6`) |
| Test-suite | 98 passed (~27 s) |
| Pre-existing Burgers comparison | sbo ≈ scikit-optimize (SBO-E04) |

## Appendix B — Measurements

**Sanity benchmark** (noise-free; same LHS initial design per seed for every method; regret = best-so-far − f\*; median [min, max]).
Branin on its standard domain, 30 calls (`n_init=5`), 10 seeds:

| method | after 10 calls | after 20 | after 30 | time/run |
|---|---|---|---|---|
| random (LHS) | 5.004 [1.52, 19.59] | 1.361 [0.59, 10.32] | 1.175 [0.07, 3.03] | — |
| sbo default (random acq.) | 1.425 [0.10, 9.80] | 0.906 [0.09, 1.85] | 0.049 [0.003, 0.199] | 0.2 s |
| sbo `refined` acq. | 1.785 [0.12, 9.80] | 0.401 [0.005, 1.63] | 0.028 [0.003, 0.112] | 1.2 s |
| sbo `refined` + ADAM (30 steps; 29 f-calls, **360 gradient calls**) | 0.066 [0.02, 0.27] | 0.024 [0.008, 0.089] | 0.016 [0.000, 0.033] | 0.5 s |
| scikit-optimize (Matérn-5/2) | 2.802 [0.12, 14.57] | 0.065 [0.001, 0.395] | 0.002 [0.000, 0.010] | 2.8 s |

Hartmann6, 60 calls (`n_init=10`), 5 seeds:

| method | after 20 calls | after 40 | after 60 | time/run |
|---|---|---|---|---|
| random (LHS) | 1.968 [0.90, 2.92] | 1.892 [0.90, 2.04] | 1.401 [0.90, 1.97] | — |
| sbo default (random acq.) | 1.260 [0.86, 2.20] | 0.391 [0.13, 0.55] | 0.187 [0.126, 0.268] | 0.6 s |
| sbo `refined` acq. | 0.742 [0.25, 1.18] | 0.093 [0.05, 0.13] | 0.027 [0.013, 0.048] | 6.6 s |
| sbo `refined` + ADAM (60 f-calls, **750 gradient calls**) | 0.002 [0.001, 0.005] | 0.002 | 0.002 | 3.7 s |
| scikit-optimize (Matérn-5/2) | 1.842 [1.18, 2.43] | 0.196 [0.03, 0.75] | 0.009 [0.002, 0.196] | 13.5 s |

Caveat: few seeds; different kernels (sbo isotropic RBF vs scikit-optimize's Matérn) — a sanity check, not a study.

**Coverage** (`coverage run --branch`, sbo-related tests, 31 tests): `__init__` 100 %, `core` 76 %, `persistence` 87 %, `plotting` **49 %**, `refinement` 89 %, `saving` 89 %; total 75 %.

**Static:** flake8 F/E9/C901 → 5 findings (see T04); mypy → 17 errors in sbo; docstrings on 34 of 71 public functions/classes, 0 with NumPy-style *Parameters*.

## Appendix C — Benchmark protocol to (re)produce and extend (SBO-T03)

- **Problems:** Branin (standard domain, f\*=0.397887), Hartmann6 on [0,1]⁶ (f\*=−3.32237); extend with Hartmann3, Ackley, Rosenbrock, Levy, one noisy objective and one CFD-like case.
- **Budgets/seeds used here:** Branin 30 calls (`n_init=5`) × 10 seeds; Hartmann6 60 calls (`n_init=10`) × 5 seeds. Paper-grade: ≥ 20–30 seeds, confidence intervals.
- **Fairness:** one shared LHS initial design per seed (`scipy.stats.qmc.LatinHypercube(d, seed=seed)`), passed to sbo as `X_init/y_init` and to scikit-optimize as `x0/y0`; objective calls counted by a wrapper (not by `len(res.y)`); gradient calls counted separately.
- **Methods:** LHS random search; sbo default (`OptimConfig(n_raw_samples=2000)`, default `GPConfig`, EI `xi=0.01`); sbo refined (`method="refined", global_method="random", n_raw_samples=2000, n_restarts=10`); sbo refined + ADAM (`n_steps=30, learning_rate=0.02`, iterations halved so f-calls match); scikit-optimize `gp_minimize(acq_func="EI", n_initial_points=0, x0, y0)`. Add: Sobol search; multi-start L-BFGS-B with the *same total gradient budget*; BoTorch (benchmark-only dependency, if the authors accept it); sbo with Matérn/ARD once restored (D01).
- **Metrics:** regret at 1/3, 2/3, 3/3 of the budget, wall time split into objective vs optimizer overhead (as `advanced_cases/…/solvers.py` already does), f-calls, gradient calls, and an equal-cost (f + gradient) plot.

## Appendix D — Environment and commands (this machine)

- `python` on PATH is the **Windows Store stub** — use the conda env interpreter: `C:\Users\lecomte\AppData\Local\anaconda3\envs\main_env\python.exe` (Python 3.12.12; NumPy 2.4.1; SciPy 1.16.3; Matplotlib 3.10.8; pytest 9.1.1; scikit-optimize 0.10.2; scikit-learn 1.7.1; torch 2.10 CPU; `pyRAMBO` installed **editable** → this working tree; no BoTorch, no `coverage`/`pytest-cov`). The base conda env (3.13.9) has mypy 1.17.1.
- Tests: `& "C:\Users\lecomte\AppData\Local\anaconda3\envs\main_env\python.exe" -m pytest -q -p no:cacheprovider` → 98 passed (~27 s). A `Windows fatal exception: code 0xc0000139` stack dump at the top of the output appears when the env is not activated; it is harmless DLL-probe noise.
- Coverage without touching the repo: `pip install --target <scratch> coverage`, then `PYTHONPATH=<scratch> python -m coverage run --branch --include="*/pyRAMBO/sbo/*" -m pytest tests/test_core.py tests/test_persistence.py tests/test_refinement.py tests/test_plotting.py` with `COVERAGE_FILE` pointing outside the repo.
- Reproduce "no LaTeX": run pytest with the MiKTeX directory removed from `PATH` (here `…\Programs\MiKTeX\miktex\bin\x64`).
- Packaging check without touching the repo: copy `pyproject.toml`, `README.md`, `LICENSE`, `src/` to a scratch folder; `python -m pip wheel . --no-deps --no-build-isolation -w <wheelhouse>`; `pip install --no-deps --target <site> <wheel>`; run the quickstart with `PYTHONPATH=<site>` from another directory.

## Appendix E — Sources and caveats

- JOSS docs fetched 2026-10-01: `https://joss.readthedocs.io/en/latest/submitting.html`, `…/review_criteria.html`, `…/paper.html`. JOSS policy changed recently (new required sections, AI disclosure, 6-month rule) — **re-read before submitting**.
- Repo/PyPI visibility checks on 2026-10-01 (SBO-J01, J06) are point-in-time.
- Statements about third-party packages in SBO-J03 are from memory and must be verified when writing the paper.
- Review limits: Windows only; Python 3.12 / NumPy 2.4 only; `mfbo`/`mobo` only grep-checked; benchmark sizes are small; no formal statistical testing.
- Scratch probe scripts (ephemeral, outside the repo) lived in the Claude session scratchpad; every finding above carries its repro inline instead.

## Progress log (append below, newest last)

- 2026-10-01 — review written; no source, test, example or config file was modified.
