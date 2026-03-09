"""CEGAR runner for tutorial case studies.

This repo contains multiple CEGAR implementations:
- 2D (synthetic, mountain car): Ethan/Krish-style RectPartition + worklist refinement.
- 3D (unicycle): specialized implementation under src/cegar/abstract_unicycle.

This script provides a single, docker-friendly entrypoint that:
- runs the requested case study
- prints base-pipeline-style metrics via PipelineLogger
- optionally evaluates against cached fixed-grid ground truth (2D cases)

Usage (repo root):
  python -u scripts/run_cegar.py --case synthetic
  python -u scripts/run_cegar.py --case mountain_car --method POLY --nx 25 --ny 25
  python -u scripts/run_cegar.py --case unicycle --unicycle-nx 20 --unicycle-gt-cache src/cegar/abstract_unicycle/cache/unicycle_cfg_e5336e8c1848.pkl

Docker (example):
  docker run --rm -v ${PWD}/runs:/app/runs <image> \
    python -u scripts/run_cegar.py --case synthetic --outdir runs/cegar/synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Path setup (make CEGAR code importable)
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_CEGAR_DIR = _SRC_DIR / "cegar"
_UNICYCLE_CEGAR_DIR = _CEGAR_DIR / "abstract_unicycle"

# Keep sys.path minimal to avoid module-name collisions.
# NOTE: src/cegar/abstract_unicycle contains a file named cegar.py, which would
# shadow the cegar package if we added that directory globally.
for p in (str(_SRC_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)


from utils.log_utils import PipelineLogger, timed_call
from utils.ground_truth_cache import build_gt_cache_path, load_gt_cache, save_gt_cache
from utils.model_checking_tools import SyntheticModelChecker, MountainCarModelChecker

from cegar.unknown_worklist import classify_state_space_worklist


_CASE_TO_SYSTEM_MOD_2D = {
    "synthetic": "cegar.helpers.systems.synthetic",
    "mountain_car": "cegar.helpers.systems.mountain_car",
}


def _uniform_grid_cells_2d(domain_rect, resolution: int) -> np.ndarray:
    xs = np.linspace(domain_rect.xmin, domain_rect.xmax, resolution + 1)
    ys = np.linspace(domain_rect.ymin, domain_rect.ymax, resolution + 1)

    cells = np.zeros((resolution * resolution, 4), dtype=float)
    k = 0
    for i in range(resolution):
        for j in range(resolution):
            cells[k, :] = (float(xs[i]), float(xs[i + 1]), float(ys[j]), float(ys[j + 1]))
            k += 1
    return cells


def _pick_checker_2d(case: str):
    if case == "mountain_car":
        return MountainCarModelChecker
    return SyntheticModelChecker


def _gt_mask_from_reference(gt_reference, n_cells: int) -> np.ndarray:
    """Normalize GT labels into boolean mask (True=safe/goal)."""

    def _to_bool(v) -> bool:
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v) != 0.0
        s = str(v).strip().lower()
        return s in {"goal", "pass", "passed", "safe", "sat", "satisfied", "true", "1"}

    gt_mask = np.zeros(n_cells, dtype=bool)
    if isinstance(gt_reference, dict):
        for i in range(n_cells):
            gt_mask[i] = _to_bool(gt_reference.get(i, False))
    else:
        arr = np.asarray(gt_reference).reshape(-1)
        if arr.size != n_cells:
            raise ValueError(f"GT reference length {arr.size} != #cells {n_cells}")
        for i in range(n_cells):
            gt_mask[i] = _to_bool(arr[i])
    return gt_mask


def _evaluate_against_gt_2d(*, absys, case: str, gt_grid_resolution: int, gt_max_steps: int, cache_dir: Path):
    Checker = _pick_checker_2d(case)
    checker = Checker(absys.system)

    d = absys.part.domain
    domain_arr = np.array([d.xmin, d.xmax, d.ymin, d.ymax], dtype=float)

    cfg = {
        "domain": [float(d.xmin), float(d.xmax), float(d.ymin), float(d.ymax)],
        "gt_grid_resolution": int(gt_grid_resolution),
        "gt_max_steps": int(gt_max_steps),
    }
    cache_path = build_gt_cache_path(str(cache_dir), case, cfg)

    if cache_path.exists():
        gt_regions = load_gt_cache(cache_path)
        loaded_cache = True
    else:
        gt_regions = checker.get_gt_reach_regions(domain_arr, gt_grid_resolution, gt_max_steps)
        save_gt_cache(cache_path, gt_regions)
        loaded_cache = False

    cells = _uniform_grid_cells_2d(d, gt_grid_resolution)
    gt_reference = checker.check_ground_truth_fast(cells, domain_arr, gt_regions)
    gt_mask = _gt_mask_from_reference(gt_reference, len(cells))

    # Build Kripke from the refined abstraction and compute sat set.
    _mc_checker, kripke, kstats, uid_to_idx, _idx_to_uid, _abs_cells, transition_map = absys.build_kripke(Checker=Checker)
    sat = {int(s) for s in checker.model_check_kripke(kripke)}

    res = int(gt_grid_resolution)
    dx = float((d.xmax - d.xmin) / res)
    dy = float((d.ymax - d.ymin) / res)

    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # uniform grid ordering matches _uniform_grid_cells_2d: k = i*res + j
    def _gt_safe(i: int, j: int) -> bool:
        return bool(gt_mask[i * res + j])

    def _gt_cell_bounds(i: int, j: int):
        xmin = d.xmin + i * dx
        xmax = d.xmin + (i + 1) * dx
        ymin = d.ymin + j * dy
        ymax = d.ymin + (j + 1) * dy
        return xmin, xmax, ymin, ymax

    def _intersects(axmin, axmax, aymin, aymax, bxmin, bxmax, bymin, bymax) -> bool:
        return not (axmax < bxmin or bxmax < axmin or aymax < bymin or bymax < aymin)

    def overlapped_gt_index_range(r):
        ix0 = int(np.floor((r.xmin - d.xmin) / dx))
        ix1 = int(np.floor((r.xmax - d.xmin) / dx))
        iy0 = int(np.floor((r.ymin - d.ymin) / dy))
        iy1 = int(np.floor((r.ymax - d.ymin) / dy))

        ix0 = _clamp(ix0, 0, res - 1)
        ix1 = _clamp(ix1, 0, res - 1)
        iy0 = _clamp(iy0, 0, res - 1)
        iy1 = _clamp(iy1, 0, res - 1)
        return ix0, ix1, iy0, iy1

    def is_truly_safe_abs_cell(r) -> bool:
        ix0, ix1, iy0, iy1 = overlapped_gt_index_range(r)
        for i in range(ix0, ix1 + 1):
            for j in range(iy0, iy1 + 1):
                gxmin, gxmax, gymin, gymax = _gt_cell_bounds(i, j)
                if _intersects(r.xmin, r.xmax, r.ymin, r.ymax, gxmin, gxmax, gymin, gymax):
                    if not _gt_safe(i, j):
                        return False
        return True

    true_safe_area = float(np.count_nonzero(gt_mask)) * (dx * dy)

    abs_truly_safe_total = 0
    abs_true_pos = 0
    captured_area = 0.0

    for uid, node in absys.part.leaves.items():
        if uid == absys.OUT_UID:
            continue
        r = node.rect
        truly_safe = is_truly_safe_abs_cell(r)
        if truly_safe:
            abs_truly_safe_total += 1

        ki = uid_to_idx.get(uid)
        is_sat = (ki is not None) and (ki in sat)

        if truly_safe and is_sat:
            abs_true_pos += 1
            captured_area += float(r.width() * r.height())

    tpr = (abs_true_pos / abs_truly_safe_total) if abs_truly_safe_total > 0 else float("nan")
    fnr = 1.0 - tpr if np.isfinite(tpr) else float("nan")
    sr = (captured_area / true_safe_area) if true_safe_area > 0 else float("nan")

    return {
        "loaded_cache": loaded_cache,
        "gt_cache_path": str(cache_path),
        "gt_true_safe_area": true_safe_area,
        "abs_truly_safe_total": int(abs_truly_safe_total),
        "abs_true_pos": int(abs_true_pos),
        "tpr": float(tpr),
        "fnr": float(fnr),
        "coverage_proportion": float(sr),
        "kripke_stats": dict(kstats),
        "sat_count": int(len(sat)),
        "kripke_state_count": int(getattr(kstats, "get", lambda k, d=None: d)("n_states", len(transition_map))),
        "kripke_edge_count": int(getattr(kstats, "get", lambda k, d=None: d)("n_edges", sum(len(s) for s in transition_map))),
    }


def _transition_map_stats(transition_map, *, out_state_included: bool):
    n_total = len(transition_map)
    n_states = n_total - 1 if out_state_included and n_total > 0 else n_total
    usable = transition_map[:n_states]

    succ_counts = np.array([len(s) for s in usable], dtype=int) if usable else np.array([], dtype=int)
    self_loops = int(sum(1 for i, succ in enumerate(usable) if i in succ))

    return {
        "n_states": int(n_states),
        "n_edges": int(sum(len(s) for s in usable)),
        "avg_successors": float(succ_counts.mean()) if succ_counts.size else 0.0,
        "max_successors": int(succ_counts.max(initial=0)) if succ_counts.size else 0,
        "self_loops": int(self_loops),
        "self_loop_proportion": float(self_loops / max(1, n_states)),
    }


def run_case_2d(args, logger: PipelineLogger):
    import importlib

    case = args.case
    system_mod = _CASE_TO_SYSTEM_MOD_2D[case]
    mod = importlib.import_module(system_mod)

    logger.stage("01 CONFIG", f"CEGAR (2D) | case={case} method={args.method} nx={args.nx} ny={args.ny}")
    logger.metrics(
        "Settings",
        [
            ("budget_steps", args.budget_steps),
            ("max_steps_validator", args.max_steps_validator),
            ("min_cell_width", args.min_cell_width),
            ("min_cell_height", args.min_cell_height),
            ("max_refine_depth", args.max_refine_depth),
            ("gt_eval", (not args.no_gt)),
            ("gt_grid_resolution", args.gt_grid_resolution),
            ("gt_max_steps", args.gt_max_steps),
        ],
    )

    spec = mod.build(nx=args.nx, ny=args.ny, method=args.method)
    absys = spec["absys"]
    phi = spec["phi"]
    goal_all_fn = spec["goal_all_fn"]

    logger.stage("02 BUILD", "Worklist CEGAR classification")
    (cls, stats), cpu_dt, wall_dt = timed_call(
        classify_state_space_worklist,
        absys,
        phi,
        goal_all_fn=goal_all_fn,
        budget_steps=args.budget_steps,
        max_steps_validator=args.max_steps_validator,
        min_cell_width=args.min_cell_width,
        min_cell_height=args.min_cell_height,
        max_refine_depth=args.max_refine_depth,
        verbose_every=args.verbose_every,
    )
    logger.runtime_line(cpu_dt, wall_dt, label="CEGAR worklist")

    logger.metrics(
        "Classification",
        [
            ("Verified", len(cls.verified)),
            ("Refuted", len(cls.refuted)),
            ("Unknown", len(cls.unknown)),
            ("Total leaves", stats.get("total_leaves")),
            ("Refine ops", stats.get("refine_ops")),
            ("Ignored", stats.get("ignored")),
            ("Real cex", stats.get("real_cex")),
        ],
    )

    logger.stage("03 KRIPKE", "Build Kripke + transition stats")
    Checker = _pick_checker_2d(case)
    checker = Checker(absys.system)
    (_mc_checker, kripke, kstats, _uid_to_idx, _idx_to_uid, _cells, transition_map), kc_cpu, kc_wall = timed_call(
        absys.build_kripke,
        Checker=Checker,
    )
    logger.runtime_line(kc_cpu, kc_wall, label="Kripke creation")

    tm_stats = _transition_map_stats(transition_map, out_state_included=True)
    logger.metrics(
        "Abstraction Metrics",
        [
            ("States (excl OUT)", tm_stats["n_states"]),
            ("Edges (excl OUT)", tm_stats["n_edges"]),
            ("Avg successors", f"{tm_stats['avg_successors']:.4f}"),
            ("Max successors", tm_stats["max_successors"]),
            ("Self-loop proportion", f"{tm_stats['self_loop_proportion']:.4f}"),
        ],
    )

    logger.stage("04 VERIFY", "CTL model checking")
    (sat_states,), mc_cpu, mc_wall = timed_call(lambda: (checker.model_check_kripke(kripke),))
    sat_states = {int(s) for s in sat_states}
    sat_rate = len(sat_states) / max(1, int(kstats.get("n_states", len(transition_map))))
    logger.runtime_line(mc_cpu, mc_wall, label="CTL model checking")

    logger.metrics(
        "Verification",
        [
            ("Formula", checker.default_ctl_formula()),
            ("Sat states", len(sat_states)),
            ("Sat rate", f"{sat_rate * 100.0:.2f}%"),
        ],
    )

    gt_metrics = None
    gt_cpu = gt_wall = 0.0
    if not args.no_gt:
        logger.stage("05 GT", "Evaluate against fixed-grid ground truth")
        cache_dir = Path(args.gt_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        (gt_metrics,), gt_cpu, gt_wall = timed_call(
            lambda: (_evaluate_against_gt_2d(
                absys=absys,
                case=case,
                gt_grid_resolution=args.gt_grid_resolution,
                gt_max_steps=args.gt_max_steps,
                cache_dir=cache_dir,
            ),)
        )
        logger.runtime_line(gt_cpu, gt_wall, label="Ground-truth evaluation")

        logger.metrics(
            "GT Metrics",
            [
                ("TPR", f"{gt_metrics['tpr']:.4f}"),
                ("FNR", f"{gt_metrics['fnr']:.4f}"),
                ("SR", f"{gt_metrics['coverage_proportion']:.4f}"),
                ("GT cache", gt_metrics["gt_cache_path"]),
            ],
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = {
        "case": case,
        "method": args.method,
        "nx": args.nx,
        "ny": args.ny,
        "phi": phi,
        "classification": {
            "verified": len(cls.verified),
            "refuted": len(cls.refuted),
            "unknown": len(cls.unknown),
        },
        "worklist_stats": stats,
        "abstraction_metrics": tm_stats,
        "kripke_stats": dict(kstats),
        "gt": gt_metrics,
        "timing": {
            "cegar_cpu_s": cpu_dt,
            "cegar_wall_s": wall_dt,
            "kripke_cpu_s": kc_cpu,
            "kripke_wall_s": kc_wall,
            "mc_cpu_s": mc_cpu,
            "mc_wall_s": mc_wall,
            "gt_cpu_s": gt_cpu,
            "gt_wall_s": gt_wall,
        },
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    logger.success(f"Saved metrics: {outdir / 'metrics.json'}")


def run_case_unicycle(args, logger: PipelineLogger):
    """Run the specialized unicycle CEGAR implementation.

    This code path currently delegates to the existing script in
    src/cegar/abstract_unicycle/run_unicycle_cegar.py.

    It writes its own metrics.json; we just run it and echo the location.
    """

    logger.stage("01 CONFIG", "CEGAR (unicycle 3D) | delegated runner")
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Build argv for the delegated script.
    delegated_argv = [
        "run_unicycle_cegar.py",
        "--init-method",
        args.unicycle_init_method,
        "--refine-method",
        args.unicycle_refine_method,
        "--nx",
        str(args.unicycle_nx),
        "--ny",
        str(args.unicycle_ny),
        "--nz",
        str(args.unicycle_nz),
        "--horizon",
        str(args.unicycle_horizon),
        "--split-budget",
        str(args.unicycle_split_budget),
        "--max-iters",
        str(args.unicycle_max_iters),
        "--gt-cache",
        str(args.unicycle_gt_cache),
        "--outdir",
        str(outdir),
    ]
    if args.verbose:
        delegated_argv.append("--verbose")

    logger.metrics(
        "Settings",
        [
            ("init_method", args.unicycle_init_method),
            ("refine_method", args.unicycle_refine_method),
            ("grid", f"{args.unicycle_nx}x{args.unicycle_ny}x{args.unicycle_nz}"),
            ("horizon", args.unicycle_horizon),
            ("split_budget", args.unicycle_split_budget),
            ("max_iters", args.unicycle_max_iters),
            ("gt_cache", str(args.unicycle_gt_cache)),
        ],
    )

    # Run in a clean subprocess to avoid module-name collisions.
    # The unicycle implementation includes a local cegar.py which is intended to
    # be imported as `cegar` when running from its directory.
    cmd = [
        sys.executable,
        "-u",
        str(_UNICYCLE_CEGAR_DIR / "run_unicycle_cegar.py"),
    ] + delegated_argv[1:]

    def _run():
        env = os.environ.copy()
        # Ensure the subprocess can import `utils.*` and other repo modules.
        existing = env.get("PYTHONPATH", "")
        prefix = str(_SRC_DIR)
        if existing:
            env["PYTHONPATH"] = prefix + os.pathsep + existing
        else:
            env["PYTHONPATH"] = prefix

        subprocess.run(cmd, check=True, cwd=str(_UNICYCLE_CEGAR_DIR), env=env)

    _, cpu_dt, wall_dt = timed_call(_run)

    logger.runtime_line(cpu_dt, wall_dt, label="Unicycle CEGAR runner")

    metrics_path = outdir / "metrics.json"
    if metrics_path.exists():
        logger.success(f"Saved metrics: {metrics_path}")
    else:
        logger.warn(f"Expected metrics.json not found at: {metrics_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["synthetic", "mountain_car", "unicycle"], required=True)
    p.add_argument("--outdir", type=str, default="runs/cegar")
    p.add_argument("--verbose", action="store_true")

    # 2D CEGAR args
    p.add_argument("--method", choices=["AABB", "POLY", "SAMPLE"], default="AABB")
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--budget-steps", type=int, default=10000)
    p.add_argument("--max-steps-validator", type=int, default=200)
    p.add_argument("--min-cell-width", type=float, default=0.0005)
    p.add_argument("--min-cell-height", type=float, default=0.0005)
    p.add_argument("--max-refine-depth", type=int, default=12)
    p.add_argument("--verbose-every", type=int, default=200)

    # GT eval (2D)
    p.add_argument("--no-gt", action="store_true")
    p.add_argument("--gt-cache-dir", type=str, default="src/cegar/gt_cache")
    p.add_argument("--gt-grid-resolution", type=int, default=100)
    p.add_argument("--gt-max-steps", type=int, default=100)

    # Unicycle specialized args
    p.add_argument("--unicycle-init-method", choices=["aabb", "poly"], default="aabb")
    p.add_argument("--unicycle-refine-method", choices=["aabb"], default="aabb")
    p.add_argument("--unicycle-nx", type=int, default=20)
    p.add_argument("--unicycle-ny", type=int, default=20)
    p.add_argument("--unicycle-nz", type=int, default=20)
    p.add_argument("--unicycle-horizon", type=int, default=100)
    p.add_argument("--unicycle-split-budget", type=int, default=0)
    p.add_argument("--unicycle-max-iters", type=int, default=200)
    p.add_argument(
        "--unicycle-gt-cache",
        type=str,
        default=str(_UNICYCLE_CEGAR_DIR / "cache" / "unicycle_cfg_e5336e8c1848.pkl"),
    )

    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # Put outputs under per-case directory unless user specified differently.
    outdir = Path(args.outdir)
    if outdir.name == "cegar" or str(outdir).endswith("runs/cegar"):
        args.outdir = str(outdir / args.case)

    logger = PipelineLogger(use_color=True, show_time=True)

    if args.case in _CASE_TO_SYSTEM_MOD_2D:
        run_case_2d(args, logger)
    else:
        run_case_unicycle(args, logger)


if __name__ == "__main__":
    main()
