import time
import argparse
import numpy as np
import sys

from unknown_worklist import classify_state_space_worklist
from krish_abstraction import KrishAbstraction
from abstraction import Rect, RectPartition
from helpers.systems.synthetic import SyntheticSystem

from helpers.model_checking_tools import (
    SyntheticModelChecker,
    MountainCarModelChecker,
    UnicycleModelChecker,   
)
from helpers.ground_truth_cache import build_gt_cache_path, load_gt_cache, save_gt_cache

sys.setrecursionlimit(200_000)

def uniform_grid_cells(domain: Rect, resolution: int) -> np.ndarray:
    """
    Build a uniform resolution x resolution grid of rectangles over `domain`.
    Returns np.ndarray shape (resolution^2, 4) with [xmin, xmax, ymin, ymax].
    """
    xs = np.linspace(domain.xmin, domain.xmax, resolution + 1)
    ys = np.linspace(domain.ymin, domain.ymax, resolution + 1)

    cells = np.zeros((resolution * resolution, 4), dtype=float)
    k = 0
    for i in range(resolution):
        for j in range(resolution):
            xmin, xmax = float(xs[i]), float(xs[i + 1])
            ymin, ymax = float(ys[j]), float(ys[j + 1])
            cells[k, :] = (xmin, xmax, ymin, ymax)
            k += 1
    return cells


# helper: map point -> refined leaf uid
def find_leaf_uid(absys, x, y):
    for uid, node in absys.part.leaves.items():
        r = node.rect
        if (r.xmin <= x <= r.xmax) and (r.ymin <= y <= r.ymax):
            return uid
    return None


# build abstraction + run classification
import importlib

def run_cegar(system_mod: str, nx: int, ny: int, budget: int, method: str, max_steps: int):
    mod = importlib.import_module(system_mod)
    spec = mod.build(nx=nx, ny=ny, method=method)

    absys = spec["absys"]
    phi = spec["phi"]
    goal_all_fn = spec["goal_all_fn"]

    t0 = time.perf_counter()
    cls, stats = classify_state_space_worklist(
        absys,
        phi,
        goal_all_fn=goal_all_fn,
        budget_steps=budget,
        max_steps_validator=max_steps,
    )
    build_time = time.perf_counter() - t0
    return absys, cls, build_time, stats, spec


def pick_checker(case_study: str):
    cs = (case_study or "").lower()
    if "mountain" in cs:
        return MountainCarModelChecker
    if "unicycle" in cs:
        return UnicycleModelChecker
    return SyntheticModelChecker


def compute_ground_truth(absys, case_study: str, resolution: int, max_steps: int, *, cache_dir="gt_cache"):
    Checker = pick_checker(case_study)
    checker = Checker(absys.system)

    d = absys.part.domain
    domain_arr = np.array([d.xmin, d.xmax, d.ymin, d.ymax], dtype=float)

    cfg = {
        "domain": [d.xmin, d.xmax, d.ymin, d.ymax],
        "gt_grid_resolution": int(resolution),
        "gt_max_steps": int(max_steps),
    }
    cache_path = build_gt_cache_path(cache_dir, case_study, cfg)

    if cache_path.exists():
        return load_gt_cache(cache_path)

    gt_regions = checker.get_gt_reach_regions(domain_arr, resolution, max_steps)
    save_gt_cache(cache_path, gt_regions)
    return gt_regions


# compare classification vs ground truth
def evaluate(absys, cls, spec, gt_resolution, gt_max_steps):
    Checker = pick_checker(spec.get("case_study", "synthetic"))
    checker = Checker(absys.system)

    d = absys.part.domain
    domain_arr = np.array([d.xmin, d.xmax, d.ymin, d.ymax], dtype=float)

    # --- ground truth ---
    t0 = time.process_time()
    case_study = spec.get("case_study", "synthetic")
    gt_regions = compute_ground_truth(absys, case_study, gt_resolution, gt_max_steps)

    # uniform grid used by Krish
    cells = uniform_grid_cells(d, gt_resolution)

    gt_reference = checker.check_ground_truth_fast(
        cells,
        domain_arr,
        gt_regions,
    )

    def _to_bool(v) -> bool:
        # numbers / bools
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v) != 0.0

        s = str(v).strip().lower()

        return s in {
            "goal", "pass", "passed", 
            "safe", "sat", "satisfied"
        }

    gt_mask = np.zeros(len(cells), dtype=bool)

    if isinstance(gt_reference, dict):
        # keys are 0..N-1 (your debug confirms ints)
        for i in range(len(cells)):
            gt_mask[i] = _to_bool(gt_reference.get(i, False))
    else:
        arr = np.asarray(gt_reference).reshape(-1)
        if arr.size != len(cells):
            raise ValueError(f"GT reference length {arr.size} != #cells {len(cells)}")
        for i in range(len(cells)):
            gt_mask[i] = _to_bool(arr[i])

    _mc_checker, kripke, _kstats, uid_to_idx, idx_to_uid, _abs_cells, transition_map = absys.build_kripke(Checker=Checker)

    nK = len(transition_map)
    K_states = set(range(nK))

    def labels_of(i: int) -> set:
        if hasattr(kripke, "L") and callable(getattr(kripke, "L")):
            return set(map(str, kripke.L(i)))
        if hasattr(kripke, "labels"):
            lab = getattr(kripke, "labels")
            if callable(lab):
                return set(map(str, lab(i)))
            if hasattr(lab, "get"):
                return set(map(str, lab.get(i, set())))
        if hasattr(kripke, "label"):
            lab = getattr(kripke, "label")
            if callable(lab):
                return set(map(str, lab(i)))
            if hasattr(lab, "get"):
                return set(map(str, lab.get(i, set())))
        return set()

    goal = {i for i in K_states if "goal" in labels_of(i)}
    safe = {i for i in K_states if "safe" in labels_of(i)}

    Z = set(goal)
    changed = True
    while changed:
        changed = False
        for s in range(nK):
            if s in Z:
                continue
            if s not in safe:
                continue
            succ = transition_map[s]
            if all((t in Z) for t in succ):
                Z.add(s)
                changed = True

    sat_kripke = Z
    print("[DEBUG] |goal| =", len(goal), "|safe| =", len(safe), "|states| =", nK, "|sat| =", len(sat_kripke))

    import math

    res = gt_resolution
    dx = (d.xmax - d.xmin) / res
    dy = (d.ymax - d.ymin) / res

    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # uniform_grid_cells order: k increments with i (x) outer, j (y) inner => k = i*res + j
    def gt_safe(i: int, j: int) -> bool:
        return bool(gt_mask[i * res + j])

    def gt_cell_bounds(i: int, j: int):
        xmin = d.xmin + i * dx
        xmax = d.xmin + (i + 1) * dx
        ymin = d.ymin + j * dy
        ymax = d.ymin + (j + 1) * dy
        return xmin, xmax, ymin, ymax

    def _intersects(axmin, axmax, aymin, aymax, bxmin, bxmax, bymin, bymax) -> bool:
        return not (
            axmax < bxmin
            or bxmax < axmin
            or aymax < bymin
            or bymax < aymin
        )

    def overlapped_gt_index_range(r):
        ix0 = int(math.floor((r.xmin - d.xmin) / dx))
        ix1 = int(math.floor((r.xmax - d.xmin) / dx))
        iy0 = int(math.floor((r.ymin - d.ymin) / dy))
        iy1 = int(math.floor((r.ymax - d.ymin) / dy))

        ix0 = _clamp(ix0, 0, res - 1)
        ix1 = _clamp(ix1, 0, res - 1)
        iy0 = _clamp(iy0, 0, res - 1)
        iy1 = _clamp(iy1, 0, res - 1)
        return ix0, ix1, iy0, iy1

    def is_truly_safe_abs_cell(r) -> bool:
        ix0, ix1, iy0, iy1 = overlapped_gt_index_range(r)
        for i in range(ix0, ix1 + 1):
            for j in range(iy0, iy1 + 1):
                gxmin, gxmax, gymin, gymax = gt_cell_bounds(i, j)
                if _intersects(r.xmin, r.xmax, r.ymin, r.ymax, gxmin, gxmax, gymin, gymax):
                    if not gt_safe(i, j):
                        return False
        return True

    # Vol(union of GT safe cells)
    true_safe_volume = float(np.count_nonzero(gt_mask)) * (dx * dy)

    abs_truly_safe_total = 0
    abs_true_pos = 0
    captured_volume = 0.0

    for uid, node in absys.part.leaves.items():
        r = node.rect

        truly_safe = is_truly_safe_abs_cell(r)
        if truly_safe:
            abs_truly_safe_total += 1

        ki = uid_to_idx.get(uid, None)
        is_sat = (ki is not None) and (ki in sat_kripke)

        if truly_safe and is_sat:
            abs_true_pos += 1
            captured_volume += (r.width() * r.height())

    print("[DEBUG] ABS truly_safe_total =", abs_truly_safe_total, "ABS true_safe_sat =", abs_true_pos)

    if abs_truly_safe_total > 0:
        tpr = abs_true_pos / abs_truly_safe_total
        fnr = 1.0 - tpr
        sr = captured_volume / true_safe_volume if true_safe_volume > 0 else 0.0
    else:
        tpr = 0.0
        fnr = 1.0
        sr = 0.0

    verify_time = time.process_time() - t0
    return {
        "tpr": float(tpr),
        "fnr": float(fnr),
        "coverage_proportion": float(sr),
    }, verify_time

def self_loop_proportion_s(transition_map) -> float:
    # EXACT implementation from log_utils.py
    self_loops = sum(1 for i, succ in enumerate(transition_map) if i in succ)
    n_states = len(transition_map)
    return self_loops / n_states if n_states > 0 else 0.0


def mean_successor_count_mSucc(transition_map) -> float:
    # mSucc = (1/|X|) * sum_x |Succ(x)|
    n_states = len(transition_map)
    return (sum(len(succ) for succ in transition_map) / n_states) if n_states > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="helpers.systems.synthetic",
                        help="Python module path with build(nx, ny, method=...)")
    parser.add_argument("--method", type=str, default="POLY", choices=["POLY", "AABB"])
    parser.add_argument("--max_steps", type=int, default=100)

    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--gt_grid_resolution", type=int, default=100)
    parser.add_argument("--gt_max_steps", type=int, default=100)
    args = parser.parse_args()

    print("\n[RUNNING CEGAR BUILD]")
    absys, cls, build_time, stats, spec = run_cegar(
        args.system,
        args.nx,
        args.ny,
        args.budget,
        args.method,
        args.max_steps,
    )

    print(len(cls.verified))
    print(len(cls.refuted))
    print(len(cls.unknown))

    print("\n[GROUND TRUTH COMPARISON]")
    results, verify_time = evaluate(
        absys,
        cls,
        spec,
        args.gt_grid_resolution,
        args.gt_max_steps,
    )

    Checker = pick_checker(spec.get("case_study", "synthetic"))
    _checker, _kripke, _stats, _uid_to_idx, _idx_to_uid, _cells, transition_map = absys.build_kripke(Checker=Checker)
    s = self_loop_proportion_s(transition_map)

    mSucc = mean_successor_count_mSucc(transition_map[:-1])

    n_states = len(transition_map) - 1

    # ---- metrics ----
    fnr = results["fnr"]
    tpr = results["tpr"]
    sr = results["coverage_proportion"]

    print("\n========== FINAL METRICS ==========")
    print(f"Build time:        {build_time:.4f}s")
    print(f"Verification time: {verify_time:.4f}s")
    print(f"X hat:               {n_states}")
    print(f"TPR:               {tpr:.4f}")
    print(f"FNR:               {fnr:.4f}")
    print(f"SR:                {sr:.4f}")
    print(f"Self-loop proportion (s): {s:.4f}")
    print(f"mSucc:                {mSucc:.4f}")
    print("===================================\n")


if __name__ == "__main__":
    main()
