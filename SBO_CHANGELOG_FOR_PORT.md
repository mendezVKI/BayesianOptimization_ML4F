# sbo → mobo / mfbo port checklist

Purpose: sbo is the reference implementation. Once sbo is finalised, replay every item below in `src/pyRAMBO/mobo` and `src/pyRAMBO/mfbo`.
Files are named by symbol, not line number. Paths are relative to `src/pyRAMBO/<pkg>/`.
Rule: **append a new numbered entry here whenever a major sbo change lands.** Mark ported items `[x mobo] [x mfbo]`.

Status legend: `[ ]` not ported, `[x]` ported, `[n/a]` not applicable.

---

## 0. Already done in mobo / mfbo (nothing to redo)
- Package layout `src/{sbo,mfbo,mobo}` → `src/pyRAMBO/{sbo,mfbo,mobo}` (plain move; `pyRAMBO/__init__.py` imports the 3 subpackages).
- `pyproject.toml`: name `pyRAMBO`, `package-dir = {"" = "src"}`, packages `pyRAMBO, pyRAMBO.sbo, .mfbo, .mobo`.
- Docstring headers and the tests' imports (`from pyRAMBO import sbo as bo`, `from pyRAMBO.sbo import persistence`).

---

## 1. Run folder layout  `[x mobo] [x mfbo]`
A run always lands in its own subfolder of `out_path`:
```
<out_path>/<timestamp | run_<n>>/
    log.log            # run start/end only
    plots/  (GIF/, conv.png, hist.png)
    res/    (meta.json, trace.npz, trace.csv, snapshots/)
```

## 2. saving.py (rewrite; sbo/saving.py is a drop-in template)  `[x mobo] [x mfbo]`
- `setup_experiment_folder(save_cfg)` now returns `(exp_path, resolved_cfg)`.
  - Always creates exactly one run subfolder (timestamp if `create_timestamp` else auto-increment `run_<n>` via `_next_run_name`).
  - Never mutates the caller's `SaveConfig`; returns `dataclasses.replace(save_cfg, out_path=exp_path)`.
  - `os.makedirs(exp_path, exist_ok=False)` for the run folder.
- `setup_logger(log_file_path)` takes the full file path (no `filename=` arg, no dir creation of `log_path` as a dir).
- `log_update` **removed**, replaced by `log_summary(logger, best_x, best_y, n_evals)` (one line, called once at the end).
  - mobo: report the Pareto front size / hypervolume instead of a scalar `best_y`.
  - mfbo: report best high-fidelity value.
- Add `import re`, `from dataclasses import replace`, `Tuple`; TYPE_CHECKING import becomes `Array, SaveConfig`.

## 3. core.py  `[x mobo] [x mfbo]`
- Import: `from .saving import setup_experiment_folder, setup_logger, log_summary`.
- `GPConfig.optimize_hyperparams` default `False` → `True`.
- `SaveConfig`: comments only (run always goes to a subfolder; log.log is metadata-only).
- `BOResult`: new field `out_path: Optional[str] = None`; set `out_path=experiment_path` in the return.
- In `bayesian_optimization`:
  - `experiment_path, save_cfg = setup_experiment_folder(save_cfg)` (rebinding: downstream plotting/persistence use the resolved cfg).
  - Logger: build `log_path = os.path.join(experiment_path, save_cfg.log_filename)`; **delete** all the per-config `logger.info(...)` lines and the table header; keep one line `BO run started: {n_iter} iterations. Configuration in meta.json.`
  - `res_path = os.path.join(experiment_path, "res")` (None if no experiment_path). `trace_path`, `snapshot_dir` and `write_meta(...)` all use `res_path` instead of `experiment_path`.
  - `trace.append(..., l_c=gp.l_c, sigma_f=gp.sigma_f, sigma_y=gp.sigma_y, n_added=len(round_y_array))`. mobo/mfbo have several GPs / fidelities: store the natural analogue (per-output arrays or per-fidelity values), keep the CSV flat.
  - **Delete** the per-iteration `log_update(logger, state)` block; add `log_summary(logger, best_x, best_y, n_evals=len(y))` right before the final `return`.

## 4. persistence.py  `[x mobo] [x mfbo]`
- `TraceLog`: new fields `l_c, sigma_f, sigma_y, n_added` (fields, `append` args, `to_arrays`).
- `TraceLog.save(path)` writes `trace.npz` **and** sibling `trace.csv` (`_write_csv`: one row per iteration, `x_next`/`x_best` split into `x_next_<j>` / `x_best_<j>` columns). `import csv`.
- `load_trace`: backward compatible with old npz (missing `l_c/sigma_f/sigma_y` → NaN, `n_added` → 0).
- `write_meta`: begins with `os.makedirs(out_path, exist_ok=True)` (it is now the first thing written, receives `res_path`).
- `load_run(out_path, ...)`: `out_path` is the run folder; reads from `out_path/res/` (`meta.json`, trace, snapshots).

## 5. plotting.py  `[x mobo] [x mfbo]`
- All figures go under `<out_path>/plots/`: GIF frames → `plots/GIF/it_XXX.png`, `plots/conv.png`, `plots/hist.png` (`os.makedirs(plots_path, exist_ok=True)`).
- Removed a stray debug `print("feff")` in `plt_state_1D`.

## 6. Tests (already migrated for sbo; mirror for the others)  `[x mobo] [x mfbo]`
- Trace path is `os.path.join(res.out_path, "res", "trace.npz")` (`res.out_path` is the resolved run folder), not `out_path/trace.npz`.
- Use `persistence.load_run(res.out_path)` rather than the raw `out_path` passed in.

---

## 7. Best-so-far no longer lags one iteration (was P1)  `[x mobo] [x mfbo]`
Row `it` of state/trace/history now records the best **including** the points evaluated at `it` (running minimum, as skopt/GPyOpt/bayes_opt). The acquisition still uses the incumbent from before the update.
In `bayesian_optimization` (core.py):
- Loop top: rename the pre-update incumbent `x_best/y_best` → `x_best_acq/y_best_acq`; use it for `make_acquisition(...)` and for `plt_state(x_best=..., y_best=...)`.
- Right after `X = vstack(...)` / `y = concatenate(...)`: recompute `best_idx = argmin(y)`, `y_best`, `x_best` from the updated data.
- `BOState`: new field `y_best_acq` (pre-update); `y_best` is now post-update. Pass `y_best_acq=y_best_acq` when building the state.
- `trace.append(x_best=x_best, y_best=y_best)` and `history["best_x"/"best_y"]` keep their names and now take the post-update values (`plt_conv` is fixed by this alone).
- Test (`test_persistence.py::test_trace_best_includes_current_iteration`): `trace.y_best == running min of [init best, y_next...]`, `y_best[-1] == res.best_y`, `state.y_best_acq >= state.y_best`.
- mobo: "best" = hypervolume/Pareto set after the update. mfbo: best high-fidelity observation after the update.
- Caveat: `y_next` is still only the first proposal; the best may come from another point of the round (`round_y`).

## 8. Gradient refinement extracted to `refinement.py`  `[n/a mobo] [n/a mfbo]` (neither package has gradient refinement)
New module `sbo/refinement.py` (no runtime import of core; `TYPE_CHECKING` for `Array`, `Bounds`). Copy it, then in core.py:
- Delete `GradientRefinementConfig`, `adam_refine_candidate`, the per-candidate refinement loop and the 9 `if refinement_cfg...: raise` lines; add `from .refinement import GradientRefinementConfig, adam_refine_candidate, refine_candidates, validate_refinement_config`.
- Validation → `validate_refinement_config(refinement_cfg, gradient)`.
- Loop → `r = refine_candidates(x_proposed, y_proposed, f_scalar, gradient, bounds, refinement_cfg, maximize=acq_cfg.maximize)`; then `x_refined, y_refined, displacements, round_X_array, round_y_array = r.x_refined, r.y_refined, r.displacements, r.round_X, r.round_y`. Remove the now-unused `width`.
- `refine_candidates` returns a `RefinementResult` (superset of the 3 requested arrays: it also returns `round_X/round_y`, the rows to append).
- Helpers: `should_merge_pair(displacement, cfg)` (merge if not `> distance_threshold`), `handle_close_pair(...)` ("final" / "midpoint").
- `__init__.py`: import the refinement names from `.refinement`.
- Tests: copy `tests/test_refinement.py`.

## 9. `plt_state` called after evaluation/refinement  `[x mobo] [x mfbo]`
- core.py: move the `plt_state(...)` call from right after the proposals are evaluated to right after the post-update best (`x_best/y_best`) is recomputed. `gp`, `acq`, `acq_res` are still pre-update (gp is only refit next iteration), i.e. what the acquisition decided from. Pass `x_best=x_best, y_best=y_best` (post-update) plus `x_new=round_X_array, y_new=round_y_array, x_refined=x_refined, y_refined=y_refined`.
- plotting.py: `plt_state`, `plt_state_1D`, `plt_state_2D` take the 4 optional args (`x_new, y_new, x_refined, y_refined`; 2D ignores `y_*`); the dispatcher forwards them. 1D: red dots "Evaluated this iter", hollow red triangles "Refined (ADAM)", star relabelled "Best so far". 2D: same overlays in `_overlay_points`, and the shared legend gets the two extra entries (`ncol=3` when >4 handles). "Best observed" → "Best so far".
- Note: the 1D plot needs `f_true` (it plots `y_true` unconditionally) — pre-existing.

## 10. HPO no longer produces NaNs (was "Known issue")  `[x mobo] [x mfbo]`
Cause: `GPConfig.theta_bounds_log=None` made L-BFGS-B unbounded; its line search reached `exp(theta)` overflow / `l_c -> 0` (`gamma = 0.5/l_c**2`) and the marginal likelihood became NaN (crashed `cholesky` with "array must not contain infs or NaNs"; seen in `test_batch_and_gradient_refinement` and in the Burgers comparison run).
Fix in sbo core.py:
- Module constant `DEFAULT_THETA_BOUNDS_LOG` (log-space, X/y are normalized): `l_c in [1e-2, 1e1]`, `sigma_f in [1e-2, 1e1]`, `sigma_y in [1e-4, 1e0]`. `optimize_gp_hyperparams` uses it when `gp_cfg.theta_bounds_log is None`.
- `negative_log_marginal_likelihood`: `if not np.all(np.isfinite(Ky)): return 1e12` before the Cholesky.
- Test: `test_rank1_update_matches_full_refit` now sets `optimize_hyperparams=False` (with HPO on, `fit_gp` refits, so rank-one can't be used; the test compares factors at equal hyperparameters).
- Reminder: rank-one is only used in iterations WITHOUT HPO (`hpo_every>1` or `optimize_hyperparams=False`) and when n > `rank_one_threshold` (default 1000).

---

## 11. `GPConfig.normalize_X` / `normalize_y` flags  `[x mobo] [x mfbo]`
Simple booleans (default `True`, = previous behaviour), like scikit-learn's `normalize_y`.
- core.py `GPConfig`: new fields `normalize_X: bool = True`, `normalize_y: bool = True` (comment: with False, `l_c/sigma_f/sigma_y` and `theta_bounds_log` are in the data's own units).
- `build_gp_model`: if `not gp_cfg.normalize_X`, build the `NormalizationHelper` with `x_lo = zeros`, `x_hi = ones` (identity transform; persistence/snapshots then work unchanged because they store the effective x_lo/x_hi/y_mean/y_std).
- `fit_gp`: `if gp_cfg.normalize_y: gp.normalizer.fit_y(y) else: gp.normalizer.y_mean, gp.normalizer.y_std = 0.0, 1.0`.
- Test: `test_core.py::test_normalization_flags` (4 flag combinations: `Xs_norm`/`ys_norm` as expected, predictions interpolate in physical units).
- mobo: multi-output, so `normalize_y` acts per output; mfbo: per fidelity.

---

## Port notes: where mobo / mfbo deliberately differ from sbo (ported 2026-10-01, all tests pass: 91)
- **saving.py**: `log_summary` is package specific. mfbo: `(logger, best_x, best_y, n_low, n_high, total_cost)`; mobo: `(logger, hypervolume, n_pareto, n_evals, ref_point)` (no scalar incumbent).
- **TraceLog hyperparameters** (item 4): mfbo records `l_lf, sigma_lf, l_delta, sigma_delta, rho, sigma_L, sigma_H`; mobo records `length_scales (d,)`, `B (M,M)`, `sigma_n (M,)` (CSV: length_scale_j, lower triangle B_ij, sigma_n_m). `n_added` is not recorded (always 1 point per iteration).
- **Item 7**: mfbo's best is high-fidelity only (moves only when an H point was evaluated; `MFBOState.y_best_acq` = pre-update). mobo records the post-update Pareto front / hypervolume in state/trace/history; `MOBOState.hypervolume_acq` = pre-update hypervolume the acquisition used. Trace/state semantics changed accordingly: mobo trace row `it` = front after evaluating `it`.
- **Item 8**: not applicable (no gradient refinement in mfbo / mobo).
- **Item 9**: mfbo `plt_state(..., x_new, y_new, level_new)` draws the evaluated point (diamond coloured by fidelity), "Best so far (high-fid.)". mobo `plt_state(..., y_new)` draws the objective values at `x_next` in the 1D panels (2D already marks `x_next`), Pareto set is post-update. No refined points (no refinement).
- **Item 10**: mobo already had `default_theta_bounds()` and a non-finite guard (nothing to do). mfbo got `DEFAULT_THETA_BOUNDS` (7 params: rho in [-10,10], log l in [1e-2,1e1], log sigma_lf/sigma_delta in [1e-3,1e3], log sigma_L/sigma_H in [1e-6,1e1]; y is not normalized there, hence the wide amplitude range) plus a finite check on K.
- **Item 11**: mobo has `normalize_X` and `normalize_y` (per output). mfbo has **`normalize_X` only**: y is deliberately kept in physical units (rho relates the physical L/H outputs, see `NormalizationHelper`).
- **Defaults**: `optimize_hyperparams` is now `True` in all three packages. Tests relying on fixed hyperparameters now set `optimize_hyperparams=False` explicitly (mfbo `test_variance_reduction_matches_full_refit`, mobo `test_icm_gp_covariance_diagonal_matches_variance`).
- **Tests added**: mfbo `test_trace_best_includes_current_iteration`, `test_normalize_X_flag`, `test_hpo_never_returns_non_finite`; mobo `test_trace_hypervolume_includes_current_iteration`, `test_normalization_flags`.

---


# ████████████████████████████████████████████████████████████████████████
# ██  MILESTONE 2026-10-01 — sbo, mobo and mfbo IN SYNC up to item 11.  ██
# ██  Everything BELOW this line is a NEW sbo-only change that has NOT  ██
# ██  been ported yet to mobo / mfbo (statuses: `[ ] mobo [ ] mfbo`).   ██
# ████████████████████████████████████████████████████████████████████████

## NEW (sbo only, to port) — items 12+

### 12. `SaveConfig.show_plots` flag: save-only by default  `[ ] mobo  [ ] mfbo`  🆕
Avoids one matplotlib pop-up per iteration on long runs (and the memory build-up of open figures).
- `SaveConfig`: new field `show_plots: bool = False` (comment: False = enabled plots are only saved under `<run>/plots/` and closed right away; True = also `plt.show()`).
- plotting.py: new helper `_finish_figure(save_cfg)` = `if save_cfg.show_plots: plt.show()` then `plt.close()`. Replace EVERY `plt.show()` (+ `plt.close()`) at the end of `plt_state_1D`, `plt_state_2D`, `plt_conv`, `plt_hist` by `_finish_figure(save_cfg)`. (Before, `plt_conv`/`plt_hist` never closed their figure at all.)
- mobo: also `plt_pareto`, `plt_state_1D/2D`, `plt_conv`, `plt_hist`; mfbo: `plt_state_1D`, `plt_conv`, `plt_hist` (each currently ends with `plt.show()`; mobo/mfbo `_save_frame`/inline savefig stay as they are).
- Test: new `tests/test_plotting.py` (`monkeypatch` `plt.show`): default -> 0 calls and no open figure but files saved; `show_plots=True` -> one call per figure. Adapt the package name for mobo / mfbo.

### 13. conv.png / hist.png no longer cropped  `[ ] mobo  [ ] mfbo`  🆕
Cause: `plt_conv` / `plt_hist` did `plt.savefig(figname, dpi=...)` with no layout handling, so the y-label / tick labels (left) and the x-label (bottom) fell outside the canvas.
- In both functions: `plt.tight_layout()` before saving and `plt.savefig(figname, dpi=save_cfg.dpi, bbox_inches="tight")`.
- Also removed a redundant `plt.tight_layout()` in `plt_state_1D` (the figure already uses `constrained_layout=True`; it only raised a UserWarning at every iteration).
- mobo/mfbo: same fix in their `plt_conv` / `plt_hist` (they save with `plt.savefig(..., dpi=save_cfg.dpi)` too; mobo's conv uses two y axes, check the right-hand label is kept).
- Test: `tests/test_plotting.py::test_conv_and_hist_are_not_cropped` (leftmost column and bottom row of the PNG must be blank; the old code gives content at both edges).

### 14. Total run time written to log.log  `[ ] mobo  [ ] mfbo`  🆕
The final summary line of `log.log` now carries the wall time of the whole run (always computed, no flag).
- core.py: `run_start_time = time.perf_counter()` right after `rng = _rng(...)` at the top of the driver; at the end `log_summary(..., total_time=time.perf_counter() - run_start_time)`. Covers initial dataset, all iterations, final plots and persistence flushes.
- saving.py: `log_summary(..., total_time)` gains the argument and appends `| total time = <s> s` (`_format_duration`: plain seconds below a minute, `1234.56 s (0:20:34)` above; `from datetime import datetime, timedelta`).
- mfbo: `log_summary(logger, best_x, best_y, n_low, n_high, total_cost)` -> add `total_time` the same way; mobo: `log_summary(logger, hypervolume, n_pareto, n_evals, ref_point)` -> add `total_time`. Timer start: after `rng = _rng(...)` in `multi_fidelity_bayesian_optimization` / `multi_objective_bayesian_optimization`.
- Test: `test_persistence.py::test_log_records_total_run_time` (log starts with "BO run started", ends with "Finished ... total time = X s").

### 15. `make_gif(run_folder, ...)`: GIF from the saved state plots  `[ ] mobo  [ ] mfbo`  🆕
- saving.py: new `make_gif(run_folder, frames_subdir="GIF", out_name="evolution.gif", fps=2.0, last_frame_hold=1.0, loop=0)`. Reads `<run>/plots/<frames_subdir>/it_XXX.png` in iteration order (`_FRAME_RE`), writes `<run>/plots/<out_name>`, returns its path. Frames of slightly different size (bbox_inches="tight") are centred on a white canvas of the largest size. Pillow only (ships with matplotlib, no new dependency, `from PIL import Image` inside the function). Raises `FileNotFoundError` mentioning `plt_state_enabled` if there are no frames.
- `__init__.py`: `from .saving import make_gif` and add `"make_gif"` to `__all__`.
- Example: `simple_cases/sbo/main_1D_sin_large_scale.py` calls `bo.make_gif(res.out_path, fps=2)` in its last cell.
- mobo: also has `plots/GIF_pareto/` (objective-space frames from `plt_pareto`): call `make_gif(run, frames_subdir="GIF_pareto", out_name="pareto_evolution.gif")` (the function already supports it). mfbo: same as sbo (`plots/GIF/`).
- Tests (`tests/test_plotting.py`): `test_make_gif_assembles_the_saved_frames` (frame count == iterations, loops forever, uniform size), `test_make_gif_without_frames_raises_a_clear_error`.

## Change log (append below, newest last)
- 2026-09-29 — items 1–6 above (folder layout, saving.py rewrite, trace.csv + GP hypers in trace, `res/` and `plots/` split, `BOResult.out_path`, `log.log` metadata-only, `optimize_hyperparams=True` default).
- 2026-09-30 — audit of best-so-far semantics; item 7 implemented in sbo only (core.py + new test).
- 2026-09-30 — item 8 (refinement.py extraction, sbo only).
- 2026-09-30 — item 9 (plt_state after refinement, sbo only).
- 2026-10-01 — item 10 (default HPO bounds + non-finite guard), found while running advanced_cases/burgers_control (comparison sbo vs skopt).
- 2026-10-01 — item 11 (normalize_X / normalize_y flags); Burgers comparison reduced to sbo vs scikit-optimize with identical settings (rank-one comparison dropped: n too small).
- 2026-10-01 — items 1-7, 9-11 ported to mobo and mfbo (item 8 n/a); see Port notes. Full test suite: 91 passed.
- 2026-10-01 — **MILESTONE**: sbo/mobo/mfbo in sync up to item 11. New sbo-only items (12+) are listed in the section just above this change log.
- 2026-10-01 — items 12 (show_plots flag) and 13 (conv/hist cropping) in sbo only — first items after the milestone.
- 2026-10-01 — item 14 (total run time in log.log), sbo only.
- 2026-10-01 — item 15 (make_gif), sbo only.
