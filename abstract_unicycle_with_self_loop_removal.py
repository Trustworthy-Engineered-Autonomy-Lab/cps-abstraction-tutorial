"""
Unicycle abstraction pipeline script.

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
from helpers.model_checking_tools import UnicycleModelChecker
from helpers.partitioning import (
    compute_transitions_AABB,
    compute_transitions_poly,
    compute_transitions_sample,
    generate_grid,
)
from helpers.plotting_3d import (
    plot_false_negative_theta_projection,
    plot_false_negative_theta_slices,
    plot_transition_comparison_theta_projection,
    plot_transition_comparison_theta_slices,
    plot_transition_theta_projection,
    plot_transition_theta_slices,
)
from helpers.systems.unicycle import UnicycleSystem
from helpers.self_loop_uni import *

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
    return (
        f"[{cell[0]:.3f}, {cell[1]:.3f}] x "
        f"[{cell[2]:.3f}, {cell[3]:.3f}] x "
        f"[{cell[4]:.3f}, {cell[5]:.3f}]"
    )

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
    out_dir = Path('out') / 'unicycle'
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path('cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run configuration ----
    ARGS = {
        # System Configuration
        'n1': 60,
        'n2': 60,
        'n3': 60,
        'x_space': (0.0, 50.0),
        'y_space': (0.0, 40.0),
        'theta_space': (-np.pi, np.pi),
        'goal_center': (40.0, 20.0),
        'goal_radius': 8.0,
        'obstacle_centers': ((25.0, 25.0),),
        'obstacle_radii': (5.0,),

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
        'plot_transitions': True,
        'plot_false_negative_map': True,
        'plot_theta_max_slices': 6,
        'plot_theta_reduce': 'mean',
        'plot_fn_projection_metric': 'fn_fraction',
    }

    n1 = ARGS['n1']
    n2 = ARGS['n2']
    n3 = ARGS['n3']
    x_min, x_max = ARGS['x_space']
    y_min, y_max = ARGS['y_space']
    theta_min, theta_max = ARGS['theta_space']
    domain_ranges = [(x_min, x_max), (y_min, y_max), (theta_min, theta_max)]
    cells_per_dim = [n1, n2, n3]

    # ---- Stage 02: grid generation ----
    logger.stage(STAGE_GRID, f'Generating {n1}x{n2}x{n3} uniform grid abstraction')
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
            ('Grid Shape', f"({n1}, {n2}, {n3})"),
            ('Bounds X', f"[{x_min}, {x_max}]"),
            ('Bounds Y', f"[{y_min}, {y_max}]"),
            ('Bounds Theta', f"[{theta_min}, {theta_max}]"),
            ('First Cell', _format_cell(cells[0])),
            ('Last Cell', _format_cell(cells[-1])),
        ],
    )
    logger.runtime_line(grid_cpu_dt, grid_wall_dt, label='Grid generation')

    system = UnicycleSystem()
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
            periodic_theta=True,
        )
        logger.record_runtime(STAGE_TRANSITIONS, 'AABB transition build', ta_cpu_dt, ta_wall_dt)
        transition_runtime_lines.append(('AABB transition build', ta_cpu_dt, ta_wall_dt))

        transition_summary_blocks.append(
            ('Transition Summary - AABB (raw)', summarize_transition_map(transition_results['AABB_raw']))
        )

        # Method 1
        aabb_shrink_working = [set(s) for s in transition_results["AABB_raw"]]
        if ARGS.get("refine_aabb_self_loops", False):
            (aabb_shrink_working, sl_stats), sl_cpu_dt, sl_wall_dt = timed_call(
                sl3_refine_self_loops_by_shrink,
                aabb_shrink_working,
                cells,
                system,
                max_steps=ARGS.get("aabb_self_loop_max_steps", 25),
                eps=ARGS.get("aabb_self_loop_eps", 1e-12),
                volume_tol=ARGS.get("aabb_self_loop_volume_tol", 0.0),
                angle_dim=2,  # theta index
                verbose=ARGS.get("aabb_self_loop_verbose", False),
            )
            logger.record_runtime(STAGE_TRANSITIONS, 'AABB shrink self-loop refinement', sl_cpu_dt, sl_wall_dt)
            logger.metrics('AABB Shrink Self-loop Refinement', [
                ('Candidates', sl_stats['candidates']),
                ('Removed', sl_stats['removed']),
                ('Kept', sl_stats['kept']),
                ('Stalled', sl_stats['stalled']),
            ])
            logger.runtime_line(sl_cpu_dt, sl_wall_dt, label='AABB shrink self-loop refinement')

        transition_results["AABB_refined_shrink"] = aabb_shrink_working
        transition_summary_blocks.append(
            ('Transition Summary - AABB (refined shrink)', summarize_transition_map(transition_results['AABB_refined_shrink']))
        )

        # Method 2
        aabb_sample_working = [set(s) for s in transition_results["AABB_raw"]]
        if ARGS.get("refine_aabb_self_loops_sample_exit", False):
            (aabb_sample_working, s_stats), s_cpu_dt, s_wall_dt = timed_call(
                sl3_refine_self_loops_by_sample_exit,
                aabb_sample_working,
                cells,
                system,
                n_samples=ARGS.get("aabb_sample_exit_n", 512),
                max_steps=ARGS.get("aabb_sample_exit_max_steps", 50),
                seed=ARGS.get("aabb_sample_exit_seed", 0),
                angle_dim=2,
                verbose=ARGS.get("aabb_sample_exit_verbose", False),
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

        transition_results["AABB_refined_sample_exit"] = aabb_sample_working
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
            tol=ARGS.get("poly_tol", 1e-9),
            periodic_theta=True,
        )
        logger.record_runtime(STAGE_TRANSITIONS, 'POLY transition build', tp_cpu_dt, tp_wall_dt)
        transition_runtime_lines.append(('POLY transition build', tp_cpu_dt, tp_wall_dt))

        transition_summary_blocks.append(
            ('Transition Summary - POLY (raw)', summarize_transition_map(transition_results['POLY_raw']))
        )

        poly_shrink_working = [set(s) for s in transition_results["POLY_raw"]]
        if ARGS.get("refine_poly_self_loops", False):
            (poly_shrink_working, sl_stats), sl_cpu_dt, sl_wall_dt = timed_call(
                sl3_refine_self_loops_by_shrink,
                poly_shrink_working,
                cells,
                system,
                max_steps=ARGS.get("poly_self_loop_max_steps", 25),
                eps=ARGS.get("poly_self_loop_eps", 1e-12),
                volume_tol=ARGS.get("poly_self_loop_volume_tol", 0.0),
                angle_dim=2,
                verbose=ARGS.get("poly_self_loop_verbose", False),
            )
            logger.record_runtime(STAGE_TRANSITIONS, 'POLY shrink self-loop refinement', sl_cpu_dt, sl_wall_dt)
            logger.metrics('POLY Shrink Self-loop Refinement', [
                ('Candidates', sl_stats['candidates']),
                ('Removed', sl_stats['removed']),
                ('Kept', sl_stats['kept']),
                ('Stalled', sl_stats['stalled']),
            ])
            logger.runtime_line(sl_cpu_dt, sl_wall_dt, label='POLY shrink self-loop refinement')

        transition_results["POLY_refined_shrink"] = poly_shrink_working
        transition_summary_blocks.append(
            ('Transition Summary - POLY (refined shrink)', summarize_transition_map(transition_results['POLY_refined_shrink']))
        )

        poly_sample_working = [set(s) for s in transition_results["POLY_raw"]]
        if ARGS.get("refine_poly_self_loops_sample_exit", False):
            (poly_sample_working, s_stats), s_cpu_dt, s_wall_dt = timed_call(
                sl3_refine_self_loops_by_sample_exit,
                poly_sample_working,
                cells,
                system,
                n_samples=ARGS.get("poly_sample_exit_n", 512),
                max_steps=ARGS.get("poly_sample_exit_max_steps", 50),
                seed=ARGS.get("poly_sample_exit_seed", 0),
                angle_dim=2,
                verbose=ARGS.get("poly_sample_exit_verbose", False),
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

        transition_results["POLY_refined_sample_exit"] = poly_sample_working
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
            periodic_theta=True,
        )
        logger.record_runtime(STAGE_TRANSITIONS, 'SAMPLE transition build', ts_cpu_dt, ts_wall_dt)
        transition_runtime_lines.append(('SAMPLE transition build', ts_cpu_dt, ts_wall_dt))

        transition_summary_blocks.append(
            ('Transition Summary - SAMPLE (raw)', summarize_transition_map(transition_results['SAMPLE_raw']))
        )

        sample_shrink_working = [set(s) for s in transition_results["SAMPLE_raw"]]
        if ARGS.get("refine_sample_self_loops", False):
            (sample_shrink_working, sl_stats), sl_cpu_dt, sl_wall_dt = timed_call(
                sl3_refine_self_loops_by_shrink,
                sample_shrink_working,
                cells,
                system,
                max_steps=ARGS.get("sample_self_loop_max_steps", 25),
                eps=ARGS.get("sample_self_loop_eps", 1e-12),
                volume_tol=ARGS.get("sample_self_loop_volume_tol", 0.0),
                angle_dim=2,
                verbose=ARGS.get("sample_self_loop_verbose", False),
            )
            logger.record_runtime(STAGE_TRANSITIONS, 'SAMPLE shrink self-loop refinement', sl_cpu_dt, sl_wall_dt)
            logger.metrics('SAMPLE Shrink Self-loop Refinement', [
                ('Candidates', sl_stats['candidates']),
                ('Removed', sl_stats['removed']),
                ('Kept', sl_stats['kept']),
                ('Stalled', sl_stats['stalled']),
            ])
            logger.runtime_line(sl_cpu_dt, sl_wall_dt, label='SAMPLE shrink self-loop refinement')

        transition_results["SAMPLE_refined_shrink"] = sample_shrink_working
        transition_summary_blocks.append(
            ('Transition Summary - SAMPLE (refined shrink)', summarize_transition_map(transition_results['SAMPLE_refined_shrink']))
        )

        sample_exit_working = [set(s) for s in transition_results["SAMPLE_raw"]]
        if ARGS.get("refine_sample_self_loops_sample_exit", False):
            (sample_exit_working, s_stats), s_cpu_dt, s_wall_dt = timed_call(
                sl3_refine_self_loops_by_sample_exit,
                sample_exit_working,
                cells,
                system,
                n_samples=ARGS.get("sample_exit_n", 512),
                max_steps=ARGS.get("sample_exit_max_steps", 50),
                seed=ARGS.get("sample_exit_seed", 0),
                angle_dim=2,
                verbose=ARGS.get("sample_exit_verbose", False),
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

        transition_results["SAMPLE_refined_sample_exit"] = sample_exit_working
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
        ("SAMPLE_raw", "SAMPLE_refined_sample_exit")
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
        checker = UnicycleModelChecker(
            system=system,
            goal_center=ARGS['goal_center'],
            goal_radius=ARGS['goal_radius'],
            obstacle_centers=ARGS['obstacle_centers'],
            obstacle_radii=ARGS['obstacle_radii'],
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
                        ('Fail states', kripke_stats['n_fail_states']),
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
        domain = (x_min, x_max, y_min, y_max, theta_min, theta_max)
        gt_cache_path = build_gt_cache_path(
            cache_dir=cache_dir,
            system_name='unicycle',
            cfg={
                'domain': domain,
                'gt_grid_resolution': ARGS['gt_grid_resolution'],
                'gt_max_steps': ARGS['gt_max_steps'],
                'goal_center': ARGS['goal_center'],
                'goal_radius': ARGS['goal_radius'],
                'obstacle_centers': ARGS['obstacle_centers'],
                'obstacle_radii': ARGS['obstacle_radii'],
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
    if ARGS['plot_transitions'] or ARGS['plot_false_negative_map']:
        logger.stage(STAGE_PLOTS, f'Rendering diagnostics to {out_dir}/')
        plots_saved = 0

        if ARGS['plot_transitions']:
            for method_name, transition_map in transition_results.items():
                method_slug = method_name.lower()

                slices_path = out_dir / f'transitions_{method_slug}_out_degree_theta_slices.png'
                projection_path = out_dir / f'transitions_{method_slug}_out_degree_theta_{ARGS["plot_theta_reduce"]}.png'

                plot_transition_theta_slices(
                    cells,
                    transition_map,
                    metric='out_degree',
                    max_slices=ARGS['plot_theta_max_slices'],
                    output_path=str(slices_path),
                    title=f'{method_name}: Out-Degree Theta Slices',
                    show=False,
                )
                plot_transition_theta_projection(
                    cells,
                    transition_map,
                    metric='out_degree',
                    reduce=ARGS['plot_theta_reduce'],
                    output_path=str(projection_path),
                    title=f'{method_name}: Out-Degree ({ARGS["plot_theta_reduce"]} over theta)',
                    show=False,
                )
                plots_saved += 2
                logger.info(f'Saved: {slices_path}')
                logger.info(f'Saved: {projection_path}')

            for left, right in pair_names:
                if left not in transition_results or right not in transition_results:
                    continue

                left_slug = left.lower()
                right_slug = right.lower()
                cmp_slices_path = out_dir / f'transitions_compare_{left_slug}_vs_{right_slug}_jaccard_theta_slices.png'
                cmp_projection_path = out_dir / (
                    f'transitions_compare_{left_slug}_vs_{right_slug}_jaccard_theta_{ARGS["plot_theta_reduce"]}.png'
                )

                plot_transition_comparison_theta_slices(
                    cells,
                    transition_results[left],
                    transition_results[right],
                    metric='state_jaccard',
                    max_slices=ARGS['plot_theta_max_slices'],
                    output_path=str(cmp_slices_path),
                    title=f'{left} vs {right}: State-wise Jaccard Theta Slices',
                    show=False,
                    vmin=0.0,
                    vmax=1.0,
                )
                plot_transition_comparison_theta_projection(
                    cells,
                    transition_results[left],
                    transition_results[right],
                    metric='state_jaccard',
                    reduce=ARGS['plot_theta_reduce'],
                    output_path=str(cmp_projection_path),
                    title=f'{left} vs {right}: State-wise Jaccard ({ARGS["plot_theta_reduce"]} over theta)',
                    show=False,
                    vmin=0.0,
                    vmax=1.0,
                )
                plots_saved += 2
                logger.info(f'Saved: {cmp_slices_path}')
                logger.info(f'Saved: {cmp_projection_path}')

        if ARGS['plot_false_negative_map'] and eval_results:
            for method_name, eval_metrics in eval_results.items():
                if method_name not in sat_results:
                    continue
                method_slug = method_name.lower()
                fn_slices_path = out_dir / f'false_negative_theta_slices_{method_slug}.png'
                fn_projection_path = out_dir / f'false_negative_theta_projection_{method_slug}.png'

                plot_false_negative_theta_slices(
                    cells,
                    sat_results[method_name],
                    eval_metrics['false_negative_states'],
                    max_slices=ARGS['plot_theta_max_slices'],
                    output_path=str(fn_slices_path),
                    title=f'{method_name}: TP/FN/TN Theta Slices',
                    show=False,
                )
                plot_false_negative_theta_projection(
                    cells,
                    sat_results[method_name],
                    eval_metrics['false_negative_states'],
                    metric=ARGS['plot_fn_projection_metric'],
                    output_path=str(fn_projection_path),
                    title=f'{method_name}: {ARGS["plot_fn_projection_metric"]} over theta',
                    show=False,
                )
                plots_saved += 2
                logger.info(f'Saved: {fn_slices_path}')
                logger.info(f'Saved: {fn_projection_path}')

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
