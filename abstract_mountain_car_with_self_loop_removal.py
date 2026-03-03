"""
Mountain Car abstraction pipeline script.

Stages:
1) Build grid and transition abstractions.
2) Build Kripke structures and run CTL model checking.
3) Evaluate against fixed-grid ground truth.
4) Render diagnostics and runtime summary.
"""

from pathlib import Path

import numpy as np

from helpers.ground_truth_cache import (
    build_gt_cache_path,
    load_gt_cache,
    save_gt_cache,
)
from helpers.log_utils import PipelineLogger, build_method_rollup_blocks, timed_call
from helpers.model_checking_tools import MountainCarModelChecker
from helpers.partitioning import (
    compute_transitions_AABB,
    compute_transitions_poly,
    compute_transitions_sample,
    generate_grid,
)
from helpers.plotting import (
    plot_false_negative_map,
    plot_grid_abstraction,
    plot_transition_comparison_heatmap,
    plot_transition_heatmap,
)
from helpers.systems.mountain_car import MountainCarSystem
from helpers.self_loop import *

# =============================================================================
# Transition-Map Utilities
# =============================================================================
def summarize_transition_map(transition_map):
    n_states = len(transition_map)
    successor_counts = np.array([len(s) for s in transition_map], dtype=int)
    nonempty = int(np.sum(successor_counts > 0))
    self_loops = int(sum(1 for i, succ in enumerate(transition_map) if i in succ))

    return [
        ('States', n_states),
        ('Non-empty', f"{nonempty} ({(100.0 * nonempty / max(1, n_states)):.2f}%)"),
        ('Avg successors', f"{successor_counts.mean():.4f}"),
        ('Max successors', successor_counts.max(initial=0)),
        ('Self-loops', f"{self_loops} ({(100.0 * self_loops / max(1, n_states)):.2f}%)"),
    ]


def compare_transition_maps(name_a, map_a, name_b, map_b):
    if len(map_a) != len(map_b):
        return [('Status', 'Different number of states; cannot compare.')]

    exact_state_match = sum(1 for a, b in zip(map_a, map_b) if a == b)
    total_a = sum(len(s) for s in map_a)
    total_b = sum(len(s) for s in map_b)
    inter = sum(len(a & b) for a, b in zip(map_a, map_b))
    union = sum(len(a | b) for a, b in zip(map_a, map_b))
    jaccard = (inter / union) if union > 0 else 1.0

    return [
        (
            'Exact state-wise set match',
            f"{exact_state_match}/{len(map_a)} ({(100.0 * exact_state_match / max(1, len(map_a))):.2f}%)",
        ),
        ('Edge (Transition) counts', f"{name_a}={total_a}, {name_b}={total_b}"),
        ('Edge (Transition) Jaccard', f"{jaccard:.6f}"),
    ]


def _format_cell(cell):
    return f"[{cell[0]:.4f}, {cell[1]:.4f}] x [{cell[2]:.4f}, {cell[3]:.4f}]"


# =============================================================================
# Main Pipeline
# =============================================================================
if __name__ == '__main__':
    # ---- Stage identifiers for ordered runtime reporting ----
    STAGE_GRID = '02 GRID'
    STAGE_TRANSITIONS = '03 TRANSITIONS'
    STAGE_KRIPKE = '04 KRIPKE'
    STAGE_MODEL_CHECK = '05 MODEL CHECK'
    STAGE_GROUND_TRUTH = '06 GROUND TRUTH'
    STAGE_PLOTS = '07 PLOTS'

    # ---- Stage 01: initialization/configuration ----
    logger = PipelineLogger()
    logger.stage('01 INIT', 'Starting configuration')
    out_dir = Path('out') / 'mountain_car'
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path('cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run configuration ----
    ARGS = {
        # System Configuration
        'n1': 100,
        'n2': 100,
        'x1_space': (-1.2, 0.6),
        'x2_space': (-0.07, 0.07),
        'goal_position_min': 0.5,

        # Transition Methods
        'run_aabb': True,
        'run_poly': True,
        'run_sample': True,

        # Sample Method Args
        'sample_n': 1000,
        'sample_seed': 0,

        # Shrink certificate
        "refine_aabb_self_loops": True,
        "refine_poly_self_loops": True,
        "refine_sample_self_loops": True,

        "aabb_self_loop_max_steps": 25,
        "aabb_self_loop_eps": 1e-12,
        "aabb_self_loop_volume_tol": 0.0,
        "aabb_self_loop_verbose": False,

        "poly_self_loop_max_steps": 25,
        "poly_self_loop_eps": 1e-12,
        "poly_self_loop_volume_tol": 0.0,
        "poly_self_loop_verbose": False,

        "sample_self_loop_max_steps": 25,
        "sample_self_loop_eps": 1e-12,
        "sample_self_loop_volume_tol": 0.0,
        "sample_self_loop_verbose": False,

        # Sample-exit certificate
        "refine_aabb_self_loops_sample_exit": True,
        "refine_poly_self_loops_sample_exit": True,
        "refine_sample_self_loops_sample_exit": True,

        "aabb_sample_exit_n": 512,
        "aabb_sample_exit_max_steps": 50,
        "aabb_sample_exit_seed": 0,
        "aabb_sample_exit_verbose": False,

        "poly_sample_exit_n": 512,
        "poly_sample_exit_max_steps": 50,
        "poly_sample_exit_seed": 0,
        "poly_sample_exit_verbose": False,

        "sample_exit_n": 512,
        "sample_exit_max_steps": 50,
        "sample_exit_seed": 0,
        "sample_exit_verbose": False,

        # Kripke
        'build_kripke': True,

        # Model Checking Args
        'run_model_checking': True,
        'run_ground_truth_eval': True,
        'gt_grid_resolution': 100,
        'gt_max_steps': 10000,
        'gt_verbose': False,

        # Plotting
        'plot_grid': True,
        'plot_transitions': True,
        'plot_false_negative_map': True,
    }

    n1 = ARGS['n1']
    n2 = ARGS['n2']
    x1_min, x1_max = ARGS['x1_space']
    x2_min, x2_max = ARGS['x2_space']
    domain_ranges = [(x1_min, x1_max), (x2_min, x2_max)]
    cells_per_dim = [n1, n2]

    # ---- Stage 02: grid generation ----
    logger.stage(STAGE_GRID, f'Generating {n1}x{n2} uniform grid abstraction')
    grid_ctx, grid_cpu_dt, grid_wall_dt = timed_call(
        generate_grid,
        domain_ranges,
        cells_per_dim,
    )
    cells = grid_ctx['cells']
    logger.record_runtime(STAGE_GRID, 'Grid generation', grid_cpu_dt, grid_wall_dt)
    logger.metrics(
        'Grid Summary',
        [
            ('Grid Shape', f"({n1}, {n2})"),
            ('Bounds Position', f"[{x1_min}, {x1_max}]"),
            ('Bounds Velocity', f"[{x2_min}, {x2_max}]"),
            ('First Cell', _format_cell(cells[0])),
            ('Last Cell', _format_cell(cells[-1])),
        ],
    )
    logger.runtime_line(grid_cpu_dt, grid_wall_dt, label='Grid generation')

    system = MountainCarSystem()
    transition_results = {}
    kripke_results = {}
    sat_results = {}
    eval_results = {}
    checker = None

    # ---- Stage 03: transition abstraction ----
    logger.stage(STAGE_TRANSITIONS, 'Computing transition maps')
    transition_summary_blocks = []
    transition_runtime_lines = []

    """
    -------------------------------------------AABB SUCCESSOR METHOD-------------------------------------------
    """
    if ARGS['run_aabb']:
        transition_results['AABB_raw'], ta_cpu_dt, ta_wall_dt = timed_call(
            compute_transitions_AABB,
            grid_ctx,
            system,
        )
        logger.record_runtime(STAGE_TRANSITIONS, 'AABB transition build', ta_cpu_dt, ta_wall_dt)
        transition_runtime_lines.append(('AABB transition build', ta_cpu_dt, ta_wall_dt))

        def count_self_loops(transition_map):
            return sum(1 for i, succ in enumerate(transition_map) if i in succ)

        # Method 1
        aabb_shrink_working = [set(s) for s in transition_results["AABB_raw"]]
        if ARGS.get('refine_aabb_self_loops', False):
            (aabb_shrink_working, sl_stats), sl_cpu_dt, sl_wall_dt = timed_call(
                refine_aabb_self_loops_by_shrink,
                aabb_shrink_working,
                cells,
                system,
                max_steps=ARGS.get('aabb_self_loop_max_steps', 25),
                eps=ARGS.get('aabb_self_loop_eps', 1e-12),
                area_tol=ARGS.get('aabb_self_loop_area_tol', 0.0),
                verbose=ARGS.get('aabb_self_loop_verbose', False),
            )
            logger.record_runtime(STAGE_TRANSITIONS, 'AABB shrink self-loop refinement', sl_cpu_dt, sl_wall_dt)
            logger.metrics('AABB Shrink Self-loop Refinement', [
                ('Candidates', sl_stats['candidates']),
                ('Removed', sl_stats['removed']),
                ('Kept', sl_stats['kept']),
                ('Stalled', sl_stats['stalled']),
            ])
            logger.runtime_line(sl_cpu_dt, sl_wall_dt, label='AABB shrink self-loop refinement')

        transition_results["AABB_refined_shrink"] = make_transition_map_total(aabb_shrink_working)

        # Method 2
        aabb_sample_working = [set(s) for s in transition_results["AABB_raw"]]
        if ARGS.get('refine_aabb_self_loops_sample_exit', False):
            (aabb_sample_working, s_stats), s_cpu_dt, s_wall_dt = timed_call(
                sl_refine_self_loops_by_sample_exit,
                aabb_sample_working,
                cells,
                system,
                n_samples=ARGS.get('aabb_sample_exit_n', 512),
                max_steps=ARGS.get('aabb_sample_exit_max_steps', 50),
                seed=ARGS.get('aabb_sample_exit_seed', 0),
                verbose=ARGS.get('aabb_sample_exit_verbose', False),
            )
            logger.record_runtime(STAGE_TRANSITIONS, 'AABB sample-exit self-loop refinement', s_cpu_dt, s_wall_dt)
            logger.metrics('AABB Sample-Exit Self-loop Refinement', [
                ('Candidates', s_stats['candidates']),
                ('Removed', s_stats['removed']),
                ('Kept', s_stats['kept']),
                ('n_samples', s_stats['n_samples']),
                ('max_steps', s_stats['max_steps']),
            ])
            logger.runtime_line(s_cpu_dt, s_wall_dt, label='AABB sample-exit self-loop refinement')

        transition_results["AABB_refined_sample_exit"] = make_transition_map_total(aabb_sample_working)

        # ---------------- Summaries: show all 3 ----------------
        loops_raw = count_self_loops(transition_results["AABB_raw"])
        loops_shrink = count_self_loops(transition_results["AABB_refined_shrink"])
        loops_sample = count_self_loops(transition_results["AABB_refined_sample_exit"])

        print(f"AABB self-loops (raw):        {loops_raw}")
        print(f"AABB self-loops (refined shrink):   {loops_shrink} (removed {loops_raw - loops_shrink})")
        print(f"AABB self-loops (refined sample):   {loops_sample} (removed {loops_raw - loops_sample})")

        transition_summary_blocks.append(
            ('Transition Summary - AABB (refined shrink)', summarize_transition_map(transition_results['AABB_refined_shrink']))
        )
        transition_summary_blocks.append(
            ('Transition Summary - AABB (refined sample-exit)', summarize_transition_map(transition_results['AABB_refined_sample_exit']))
        )
    """
    -------------------------------------------AABB SUCCESSOR METHOD-------------------------------------------
    """

    """
    -------------------------------------------POLY SUCCESSOR METHOD-------------------------------------------
    """
    if ARGS['run_poly']:
        transition_results['POLY_raw'], tp_cpu_dt, tp_wall_dt = timed_call(
            compute_transitions_poly,
            grid_ctx,
            system,
        )
        logger.record_runtime(STAGE_TRANSITIONS, 'POLY transition build', tp_cpu_dt, tp_wall_dt)
        transition_runtime_lines.append(('POLY transition build', tp_cpu_dt, tp_wall_dt))

        def count_self_loops(transition_map):
            return sum(1 for i, succ in enumerate(transition_map) if i in succ)

        # Method 1
        poly_shrink_working = [set(s) for s in transition_results["POLY_raw"]]
        if ARGS.get('refine_poly_self_loops', False):
            (poly_shrink_working, sl_stats), sl_cpu_dt, sl_wall_dt = timed_call(
                refine_aabb_self_loops_by_shrink,
                poly_shrink_working,
                cells,
                system,
                max_steps=ARGS.get('poly_self_loop_max_steps', 25),
                eps=ARGS.get('poly_self_loop_eps', 1e-12),
                area_tol=ARGS.get('poly_self_loop_area_tol', 0.0),
                verbose=ARGS.get('poly_self_loop_verbose', False),
            )
            logger.record_runtime(STAGE_TRANSITIONS, 'POLY shrink self-loop refinement', sl_cpu_dt, sl_wall_dt)
            logger.metrics('POLY Shrink Self-loop Refinement', [
                ('Candidates', sl_stats['candidates']),
                ('Removed', sl_stats['removed']),
                ('Kept', sl_stats['kept']),
                ('Stalled', sl_stats['stalled']),
            ])
            logger.runtime_line(sl_cpu_dt, sl_wall_dt, label='POLY shrink self-loop refinement')

        transition_results["POLY_refined_shrink"] = make_transition_map_total(poly_shrink_working)

        # Method 2
        poly_sample_working = [set(s) for s in transition_results["POLY_raw"]]
        if ARGS.get('refine_poly_self_loops_sample_exit', False):
            (poly_sample_working, s_stats), s_cpu_dt, s_wall_dt = timed_call(
                sl_refine_self_loops_by_sample_exit,
                poly_sample_working,
                cells,
                system,
                n_samples=ARGS.get('poly_sample_exit_n', 512),
                max_steps=ARGS.get('poly_sample_exit_max_steps', 50),
                seed=ARGS.get('poly_sample_exit_seed', 0),
                verbose=ARGS.get('poly_sample_exit_verbose', False),
            )
            logger.record_runtime(STAGE_TRANSITIONS, 'POLY sample-exit self-loop refinement', s_cpu_dt, s_wall_dt)
            logger.metrics('POLY Sample-Exit Self-loop Refinement', [
                ('Candidates', s_stats['candidates']),
                ('Removed', s_stats['removed']),
                ('Kept', s_stats['kept']),
                ('n_samples', s_stats['n_samples']),
                ('max_steps', s_stats['max_steps']),
            ])
            logger.runtime_line(s_cpu_dt, s_wall_dt, label='POLY sample-exit self-loop refinement')

        transition_results["POLY_refined_sample_exit"] = make_transition_map_total(poly_sample_working)

        # ---------------- Summaries: show all 3 ----------------
        loops_raw = count_self_loops(transition_results["POLY_raw"])
        loops_shrink = count_self_loops(transition_results["POLY_refined_shrink"])
        loops_sample = count_self_loops(transition_results["POLY_refined_sample_exit"])

        print(f"POLY self-loops (raw):        {loops_raw}")
        print(f"POLY self-loops (refined shrink):   {loops_shrink} (removed {loops_raw - loops_shrink})")
        print(f"POLY self-loops (refined sample):   {loops_sample} (removed {loops_raw - loops_sample})")

        transition_summary_blocks.append(
            ('Transition Summary - POLY (refined shrink)', summarize_transition_map(transition_results['POLY_refined_shrink']))
        )
        transition_summary_blocks.append(
            ('Transition Summary - POLY (refined sample-exit)', summarize_transition_map(transition_results['POLY_refined_sample_exit']))
        )
    """
    -------------------------------------------POLY SUCCESSOR METHOD-------------------------------------------
    """

    """
    -------------------------------------------SAMPLE SUCCESSOR METHOD-------------------------------------------
    """
    if ARGS['run_sample']:
        transition_results['SAMPLE_raw'], ts_cpu_dt, ts_wall_dt = timed_call(
            compute_transitions_sample,
            grid_ctx,
            system,
            batch_n_samples=ARGS['sample_n'],
            rng_seed=ARGS['sample_seed'],
        )
        logger.record_runtime(STAGE_TRANSITIONS, 'SAMPLE transition build', ts_cpu_dt, ts_wall_dt)
        transition_runtime_lines.append(('SAMPLE transition build', ts_cpu_dt, ts_wall_dt))

        def count_self_loops(transition_map):
            return sum(1 for i, succ in enumerate(transition_map) if i in succ)

        # Method 1
        sample_shrink_working = [set(s) for s in transition_results["SAMPLE_raw"]]
        if ARGS.get('refine_sample_self_loops', False):
            (sample_shrink_working, sl_stats), sl_cpu_dt, sl_wall_dt = timed_call(
                refine_aabb_self_loops_by_shrink,   # same shrink certificate
                sample_shrink_working,
                cells,
                system,
                max_steps=ARGS.get('sample_self_loop_max_steps', 25),
                eps=ARGS.get('sample_self_loop_eps', 1e-12),
                area_tol=ARGS.get('sample_self_loop_area_tol', 0.0),
                verbose=ARGS.get('sample_self_loop_verbose', False),
            )
            logger.record_runtime(STAGE_TRANSITIONS, 'SAMPLE shrink self-loop refinement', sl_cpu_dt, sl_wall_dt)
            logger.metrics('SAMPLE Shrink Self-loop Refinement', [
                ('Candidates', sl_stats['candidates']),
                ('Removed', sl_stats['removed']),
                ('Kept', sl_stats['kept']),
                ('Stalled', sl_stats['stalled']),
            ])
            logger.runtime_line(sl_cpu_dt, sl_wall_dt, label='SAMPLE shrink self-loop refinement')

        transition_results["SAMPLE_refined_shrink"] = make_transition_map_total(sample_shrink_working)

        # Method 2
        sample_exit_working = [set(s) for s in transition_results["SAMPLE_raw"]]
        if ARGS.get('refine_sample_self_loops_sample_exit', False):
            (sample_exit_working, s_stats), s_cpu_dt, s_wall_dt = timed_call(
                sl_refine_self_loops_by_sample_exit,
                sample_exit_working,
                cells,
                system,
                n_samples=ARGS.get('sample_exit_n', 512),
                max_steps=ARGS.get('sample_exit_max_steps', 50),
                seed=ARGS.get('sample_exit_seed', 0),
                verbose=ARGS.get('sample_exit_verbose', False),
            )
            logger.record_runtime(STAGE_TRANSITIONS, 'SAMPLE sample-exit self-loop refinement', s_cpu_dt, s_wall_dt)
            logger.metrics('SAMPLE Sample-Exit Self-loop Refinement', [
                ('Candidates', s_stats['candidates']),
                ('Removed', s_stats['removed']),
                ('Kept', s_stats['kept']),
                ('n_samples', s_stats['n_samples']),
                ('max_steps', s_stats['max_steps']),
            ])
            logger.runtime_line(s_cpu_dt, s_wall_dt, label='SAMPLE sample-exit self-loop refinement')

        transition_results["SAMPLE_refined_sample_exit"] = make_transition_map_total(sample_exit_working)

        # ---------------- Summaries: show all 3 ----------------
        loops_raw = count_self_loops(transition_results["SAMPLE_raw"])
        loops_shrink = count_self_loops(transition_results["SAMPLE_refined_shrink"])
        loops_sample_exit = count_self_loops(transition_results["SAMPLE_refined_sample_exit"])

        print(f"SAMPLE self-loops (raw):         {loops_raw}")
        print(f"SAMPLE self-loops (refined shrink):    {loops_shrink} (removed {loops_raw - loops_shrink})")
        print(f"SAMPLE self-loops (refined sample-exit):{loops_sample_exit} (removed {loops_raw - loops_sample_exit})")

        transition_summary_blocks.append(
            ('Transition Summary - SAMPLE (refined shrink)', summarize_transition_map(transition_results['SAMPLE_refined_shrink']))
        )
        transition_summary_blocks.append(
            ('Transition Summary - SAMPLE (refined sample-exit)', summarize_transition_map(transition_results['SAMPLE_refined_sample_exit']))
        )
    """
    -------------------------------------------SAMPLE SUCCESSOR METHOD-------------------------------------------
    """

    if transition_summary_blocks:
        logger.metrics_side_by_side(transition_summary_blocks, gap=6)
    for label, cpu_dt, wall_dt in transition_runtime_lines:
        logger.runtime_line(cpu_dt, wall_dt, label=label)

    pair_names = [
        # AABB internal comparisons
        ("AABB_raw", "AABB_refined_shrink"),
        ("AABB_raw", "AABB_refined_sample_exit"),

        # POLY internal comparisons
        ("POLY_raw", "POLY_refined_shrink"),
        ("POLY_raw", "POLY_refined_sample_exit"),

        # SAMPLE internal comparisons
        ("SAMPLE_raw", "SAMPLE_refined_shrink"),
        ("SAMPLE_raw", "SAMPLE_refined_sample_exit"),
    ]
    
    for left, right in pair_names:
        if left in transition_results and right in transition_results:
            logger.metrics(
                f'Transition Comparison - {left} vs {right}',
                compare_transition_maps(left, transition_results[left], right, transition_results[right]),
            )

    # ---- Stage 04: Kripke construction ----
    if ARGS['build_kripke'] and transition_results:
        logger.stage(STAGE_KRIPKE, 'Building Kripke structures from transition maps')
        checker = MountainCarModelChecker(
            system=system,
            goal_position_min=ARGS['goal_position_min'],
        )

        kripke_summary_blocks = []
        kripke_runtime_lines = []
        for method_name, transition_map in transition_results.items():
            (kripke, kripke_stats), kc_cpu_dt, kc_wall_dt = timed_call(
                checker.create_kripke,
                cells,
                transition_map,
            )
            kripke_results[method_name] = kripke
            logger.record_runtime(STAGE_KRIPKE, f'{method_name} Kripke creation', kc_cpu_dt, kc_wall_dt)
            kripke_summary_blocks.append(
                (
                    f'Kripke Summary - {method_name}',
                    [
                        ('States', kripke_stats['n_states']),
                        ('Edges (Transitions)', kripke_stats['n_edges']),
                        ('Goal states', kripke_stats['n_goal_states']),
                        ('OOB source states', kripke_stats['n_oob_sources']),
                        ('Avg successors', f"{kripke_stats['avg_successors']:.4f}"),
                        ('Max successors', kripke_stats['max_successors']),
                        ('Self-loops', kripke_stats['self_loops']),
                    ],
                )
            )
            kripke_runtime_lines.append((f'{method_name} Kripke creation', kc_cpu_dt, kc_wall_dt))

        if kripke_summary_blocks:
            logger.metrics_side_by_side(kripke_summary_blocks, gap=6)
        for label, cpu_dt, wall_dt in kripke_runtime_lines:
            logger.runtime_line(cpu_dt, wall_dt, label=label)

    # ---- Stage 05: CTL model checking ----
    if ARGS['run_model_checking'] and checker is not None and kripke_results:
        logger.stage(STAGE_MODEL_CHECK, 'Running CTL model checking')
        formula = checker.default_ctl_formula()
        mc_summary_blocks = []
        mc_runtime_lines = []
        for method_name, kripke in kripke_results.items():
            sat_states, mc_cpu_dt, mc_wall_dt = timed_call(checker.model_check_kripke, kripke)
            sat_results[method_name] = sat_states
            sat_rate = len(sat_states) / len(cells)
            logger.record_runtime(STAGE_MODEL_CHECK, f'{method_name} CTL model checking', mc_cpu_dt, mc_wall_dt)
            mc_summary_blocks.append(
                (
                    f'Model Checking Summary - {method_name}',
                    [
                        ('Satisfying states', len(sat_states)),
                        ('Sat rate', f"{sat_rate * 100.0:.2f}%"),
                        ('Formula', formula),
                    ],
                )
            )
            mc_runtime_lines.append((f'{method_name} CTL model checking', mc_cpu_dt, mc_wall_dt))

        if mc_summary_blocks:
            logger.metrics_side_by_side(mc_summary_blocks, gap=6)
        for label, cpu_dt, wall_dt in mc_runtime_lines:
            logger.runtime_line(cpu_dt, wall_dt, label=label)

    # ---- Stage 06: ground-truth evaluation ----
    if ARGS['run_ground_truth_eval'] and checker is not None and sat_results:
        logger.stage(STAGE_GROUND_TRUTH, 'Building ground truth and evaluating FNR/SR')
        domain = (x1_min, x1_max, x2_min, x2_max)
        gt_cache_path = build_gt_cache_path(
            cache_dir=cache_dir,
            system_name='mountain_car',
            cfg={
                'domain': domain,
                'gt_grid_resolution': ARGS['gt_grid_resolution'],
                'gt_max_steps': ARGS['gt_max_steps'],
                'goal_position_min': ARGS['goal_position_min'],
            },
        )

        gt_prep_cpu_dt = 0.0
        gt_prep_wall_dt = 0.0
        loaded_from_cache = False

        if gt_cache_path.exists():
            try:
                gt_reach_regions, gt_load_cpu_dt, gt_load_wall_dt = timed_call(
                    load_gt_cache,
                    gt_cache_path,
                )
                gt_prep_cpu_dt += gt_load_cpu_dt
                gt_prep_wall_dt += gt_load_wall_dt
                loaded_from_cache = True
                logger.record_runtime(STAGE_GROUND_TRUTH, 'Ground-truth cache load', gt_load_cpu_dt, gt_load_wall_dt)
                logger.info(f'Loaded ground-truth cache: {gt_cache_path}')
            except OSError as exc:
                logger.warn(f'Ground-truth cache load failed ({exc}); rebuilding cache.')

        if not loaded_from_cache:
            gt_reach_regions, gt_cpu_dt, gt_wall_dt = timed_call(
                checker.get_gt_reach_regions,
                domain,
                ARGS['gt_grid_resolution'],
                ARGS['gt_max_steps'],
                ARGS['gt_verbose'],
            )
            logger.record_runtime(STAGE_GROUND_TRUTH, 'Ground-truth reachability', gt_cpu_dt, gt_wall_dt)
            gt_prep_cpu_dt += gt_cpu_dt
            gt_prep_wall_dt += gt_wall_dt

            try:
                _, gt_save_cpu_dt, gt_save_wall_dt = timed_call(
                    save_gt_cache,
                    gt_cache_path,
                    gt_reach_regions,
                )
                logger.record_runtime(STAGE_GROUND_TRUTH, 'Ground-truth cache save', gt_save_cpu_dt, gt_save_wall_dt)
                logger.info(f'Saved ground-truth cache: {gt_cache_path}')
            except OSError as exc:
                logger.warn(f'Ground-truth cache save failed ({exc}); continuing without cache.')

        ground_truth_reference, gtr_cpu_dt, gtr_wall_dt = timed_call(
            checker.check_ground_truth_fast,
            cells,
            domain,
            gt_reach_regions,
        )
        logger.record_runtime(STAGE_GROUND_TRUTH, 'Ground-truth mapping', gtr_cpu_dt, gtr_wall_dt)
        gt_prep_cpu_dt += gtr_cpu_dt
        gt_prep_wall_dt += gtr_wall_dt

        dynamics_metrics, dm_cpu_dt, dm_wall_dt = timed_call(checker.compute_dynamics_metrics, cells)
        logger.record_runtime(STAGE_GROUND_TRUTH, 'Dynamics metrics', dm_cpu_dt, dm_wall_dt)

        gt_dyn_cpu_dt = gt_prep_cpu_dt + dm_cpu_dt
        gt_dyn_wall_dt = gt_prep_wall_dt + dm_wall_dt
        logger.info(
            f"Ground-truth prep: regions={len(gt_reach_regions)}, "
            f"grid_resolution={ARGS['gt_grid_resolution']}, max_steps={ARGS['gt_max_steps']}"
        )
        logger.runtime_line(gt_dyn_cpu_dt, gt_dyn_wall_dt, label='Ground-truth + dynamics')
        logger.info(
            "Dynamics metrics: "
            f"avg_cell={dynamics_metrics['avg_cell_measure']:.6f}, "
            f"avg_image={dynamics_metrics['avg_image_measure']:.6f}, "
            f"avg_ratio={dynamics_metrics['avg_image_over_cell']:.6f}, "
            f"avg_centroid_disp={dynamics_metrics['avg_centroid_displacement']:.6f}, "
            f"avg_ioa={dynamics_metrics['avg_intersection_over_area']:.6f}, "
            f"avg_iou={dynamics_metrics['avg_intersection_over_union']:.6f}"
        )

        eval_summary_blocks = []
        eval_runtime_lines = []
        for method_name, sat_states in sat_results.items():
            eval_metrics, ev_cpu_dt, ev_wall_dt = timed_call(
                checker.evaluate_against_ground_truth,
                sat_states,
                cells,
                ground_truth_reference,
                gt_reach_regions,
            )
            eval_results[method_name] = eval_metrics
            logger.record_runtime(STAGE_GROUND_TRUTH, f'{method_name} ground-truth evaluation', ev_cpu_dt, ev_wall_dt)
            eval_summary_blocks.append(
                (
                    f'Ground-Truth Evaluation - {method_name}',
                    [
                        ('Sat rate', f"{eval_metrics['sat_rate']:.6f}"),
                        ('Sat coverage', f"{eval_metrics['sat_coverage']:.6f}"),
                        ('GT goal fraction', f"{eval_metrics['gt_goal_fraction']:.6f}"),
                        ('Coverage proportion (SR)', f"{eval_metrics['coverage_proportion']:.6f}"),
                        ('True negative fraction (TNP)', f"{eval_metrics['true_negative_fraction']:.6f}"),
                        ('False negative rate (FNR)', f"{eval_metrics['fnr']:.6f}"),
                        ('Checked sat states', eval_metrics['n_checked_sat_states']),
                        ('True sat states', eval_metrics['n_true_sat_states']),
                        ('False negative states', len(eval_metrics['false_negative_states'])),
                    ],
                )
            )
            eval_runtime_lines.append((f'{method_name} ground-truth evaluation', ev_cpu_dt, ev_wall_dt))

        if eval_summary_blocks:
            logger.metrics_side_by_side(eval_summary_blocks, gap=6)
        for label, cpu_dt, wall_dt in eval_runtime_lines:
            logger.runtime_line(cpu_dt, wall_dt, label=label)

    # ---- Stage 07: diagnostics plotting ----
    if ARGS['plot_grid'] or ARGS['plot_transitions'] or ARGS['plot_false_negative_map']:
        logger.stage(STAGE_PLOTS, f'Rendering diagnostics to {out_dir}/')
        plots_saved = 0

        if ARGS['plot_grid']:
            grid_path = out_dir / 'grid_abstraction.png'
            plot_grid_abstraction(
                cells,
                (x1_min, x1_max),
                (x2_min, x2_max),
                output_path=str(grid_path),
                title='Uniform Rectilinear Abstraction (Mountain Car)',
                show=False,
                system=system,
            )
            plots_saved += 1
            logger.info(f'Saved: {grid_path}')

        if ARGS['plot_transitions']:
            for method_name, transition_map in transition_results.items():
                method_slug = method_name.lower()
                out_degree_path = out_dir / f'transitions_{method_slug}_out_degree_heatmap.png'
                self_loop_path = out_dir / f'transitions_{method_slug}_self_loop_heatmap.png'

                plot_transition_heatmap(
                    cells,
                    transition_map,
                    metric='out_degree',
                    output_path=str(out_degree_path),
                    title=f'{method_name}: Out-Degree Heatmap',
                    show=False,
                )
                plot_transition_heatmap(
                    cells,
                    transition_map,
                    metric='self_loop',
                    output_path=str(self_loop_path),
                    title=f'{method_name}: Self-Loop Heatmap',
                    show=False,
                    vmin=0.0,
                    vmax=1.0,
                )
                plots_saved += 2
                logger.info(f'Saved: {out_degree_path}')
                logger.info(f'Saved: {self_loop_path}')

            for left, right in pair_names:
                if left not in transition_results or right not in transition_results:
                    continue

                left_slug = left.lower()
                right_slug = right.lower()
                jaccard_path = out_dir / f'transitions_compare_{left_slug}_vs_{right_slug}_state_jaccard.png'
                diff_path = out_dir / f'transitions_compare_{left_slug}_vs_{right_slug}_symmetric_diff_count.png'

                plot_transition_comparison_heatmap(
                    cells,
                    transition_results[left],
                    transition_results[right],
                    metric='state_jaccard',
                    output_path=str(jaccard_path),
                    title=f'{left} vs {right}: State-wise Jaccard',
                    show=False,
                    vmin=0.0,
                    vmax=1.0,
                )
                plot_transition_comparison_heatmap(
                    cells,
                    transition_results[left],
                    transition_results[right],
                    metric='symmetric_diff_count',
                    output_path=str(diff_path),
                    title=f'{left} vs {right}: Symmetric Difference Count',
                    show=False,
                )
                plots_saved += 2
                logger.info(f'Saved: {jaccard_path}')
                logger.info(f'Saved: {diff_path}')

        if ARGS['plot_false_negative_map'] and eval_results:
            for method_name, eval_metrics in eval_results.items():
                if method_name not in sat_results:
                    continue
                method_slug = method_name.lower()
                fn_path = out_dir / f'false_negative_map_{method_slug}.png'
                plot_false_negative_map(
                    cells,
                    sat_results[method_name],
                    eval_metrics['false_negative_states'],
                    output_path=str(fn_path),
                    title=f'{method_name}: TP/FN/TN Map',
                    show=False,
                    x1_space=(x1_min, x1_max),
                    x2_space=(x2_min, x2_max),
                )
                plots_saved += 1
                logger.info(f'Saved: {fn_path}')

        if plots_saved == 0:
            logger.info('No plots generated for current flags and available data.')

    # ---- Stage 08: runtime summary ----
    logger.stage('08 RUNTIME', 'Stage-level runtime summary')
    logger.runtime_summary_by_stage()

    # ---- Stage 09: final rollup ----
    logger.stage('09 ROLLUP', 'Final method rollup')
    rollup_blocks = build_method_rollup_blocks(
        method_names=[
        k for k in (
            "AABB_raw", "AABB_refined_shrink", "AABB_refined_sample_exit",
            "POLY_raw", "POLY_refined_shrink", "POLY_refined_sample_exit",
            "SAMPLE_raw", "SAMPLE_refined_shrink", "SAMPLE_refined_sample_exit",
        )
        if k in transition_results
        ],
        transition_results=transition_results,
        eval_results=eval_results,
        runtimes_by_stage=logger._runtimes_by_stage,
        stage_grid=STAGE_GRID,
        stage_transitions=STAGE_TRANSITIONS,
        stage_kripke=STAGE_KRIPKE,
        stage_model_check=STAGE_MODEL_CHECK,
        stage_ground_truth=STAGE_GROUND_TRUTH,
    )
    if rollup_blocks:
        logger.metrics_side_by_side(rollup_blocks, gap=6)

    # ---- Stage 10: completion ----
    total_wall_dt = logger.total_wall_seconds()
    logger.stage('10 DONE', f'Grid abstraction creation complete | Total wall {total_wall_dt:.6f}s')
