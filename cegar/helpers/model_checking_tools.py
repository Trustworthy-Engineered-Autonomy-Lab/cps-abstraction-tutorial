import numpy as np
import pyModelChecking as pmc
import pyModelChecking.CTL as CTL

from helpers.math_utils import build_hyperrectangle_vertices


def _normalize_grid_resolution(grid_resolution, dims):
    """
    Normalize grid-resolution input to a tuple of edge counts per dimension.

    A scalar `r` means `r` edges in every dimension (thus `r-1` cells each).
    """
    if np.isscalar(grid_resolution):
        return (int(grid_resolution),) * dims
    return tuple(int(v) for v in grid_resolution)


def _overlap_index_range(edges, lo, hi, upper_side='left'):
    """Return fixed-grid index range [i_lo, i_hi] whose cells overlap [lo, hi]."""
    edges = np.asarray(edges, dtype=float)
    lo = float(lo)
    hi = float(hi)

    n = edges.size - 1
    i_lo = int(np.searchsorted(edges, lo, side='right') - 1)
    i_hi = int(np.searchsorted(edges, hi, side=upper_side) - 1)

    i_lo = max(0, min(n - 1, i_lo))
    i_hi = max(0, min(n - 1, i_hi))
    return i_lo, i_hi


def _infer_fixed_grid_edges(domain, gt_reach_regions, dims):
    """
    Infer fixed-grid edges from ground-truth dictionary keys and domain bounds.
    """
    domain = np.asarray(domain, dtype=float).reshape(-1)
    max_idx = [max(int(k[d]) for k in gt_reach_regions.keys()) for d in range(dims)]
    edge_counts = [m + 2 for m in max_idx]

    edges = [
        np.linspace(float(domain[2 * d]), float(domain[2 * d + 1]), edge_counts[d])
        for d in range(dims)
    ]
    return edges


class BaseModelChecker:
    """
    Base Kripke-construction and verification utility for grid abstractions.

    Subclasses provide labeling predicates by implementing `_compute_goal_mask`
    (and optionally `_compute_fail_mask` / `_compute_oob_mask`) and may add
    system-specific ground-truth helpers.
    """

    def __init__(
        self,
        system,
        include_oob_state=False,
    ):
        self.system = system
        self.include_oob_state = bool(include_oob_state)

    @staticmethod
    def _normalize_state_ids(state_ids):
        return {int(s) for s in state_ids}

    def create_kripke(self, cells, transition_map):
        """
        Build and return a pyModelChecking Kripke structure.

        Args:
            cells: (N, 2*D) array of abstract cells.
            transition_map: precomputed transitions (list[set[int]]).
            Returns (kripke, stats_dict).
        """
        n_cells = cells.shape[0]

        goal_mask = np.asarray(self._compute_goal_mask(cells), dtype=bool)
        fail_mask = np.asarray(self._compute_fail_mask(cells), dtype=bool)

        if self.include_oob_state:
            oob_mask = self._compute_oob_mask(cells)
        else:
            oob_mask = np.zeros(n_cells, dtype=bool)

        oob_state_id = n_cells if self.include_oob_state else None
        n_states = n_cells + (1 if self.include_oob_state else 0)

        states = list(range(n_states))
        initial_states = list(range(n_cells))

        labels = {}
        for i in range(n_cells):
            if goal_mask[i]:
                labels[i] = ['goal']
            elif fail_mask[i]:
                labels[i] = ['fail']
            else:
                labels[i] = ['safe']
        if self.include_oob_state:
            labels[oob_state_id] = ['fail']

        edges = []
        for src, succ in enumerate(transition_map):
            for dst in succ:
                dst = int(dst)
                edges.append((src, dst))

            if self.include_oob_state and oob_mask[src]:
                edges.append((src, oob_state_id))

        if self.include_oob_state:
            edges.append((oob_state_id, oob_state_id))

        kripke = pmc.Kripke(
            S=states,
            S0=initial_states,
            R=edges,
            L=labels,
        )

        successor_counts = np.zeros(n_cells, dtype=int)
        self_loops = 0
        for src, dst in edges:
            if src < n_cells:
                successor_counts[src] += 1
                if src == dst:
                    self_loops += 1

        stats = {
            'n_cells': n_cells,
            'n_states': n_states,
            'n_edges': len(edges),
            'avg_successors': float(successor_counts.mean()),
            'max_successors': int(successor_counts.max()),
            'self_loops': int(self_loops),
            'n_goal_states': int(np.sum(goal_mask)),
            'n_fail_states': int(np.sum(fail_mask)),
            'n_oob_sources': int(np.sum(oob_mask)),
        }
        return kripke, stats

    def default_ctl_formula(self):
        """Default CTL objective for the checker."""
        return 'A (safe U goal)'

    def model_check_kripke(self, kripke_structure):
        """
        Run CTL model checking and return satisfying state IDs.

        Args:
            kripke_structure: pyModelChecking Kripke object.
            Uses the checker's default CTL formula.
        """
        return CTL.modelcheck(kripke_structure, self.default_ctl_formula())

    @staticmethod
    def false_negative_rate(true_safe_states, checked_safe_states):
        """
        Compute false-negative rate against a ground-truth safe-state set.
        """
        true_safe_states = {int(s) for s in true_safe_states}
        checked_safe_states = {int(s) for s in checked_safe_states}
        false_negative_states = {s for s in true_safe_states if s not in checked_safe_states}
        denom = len(true_safe_states)
        fnr = (len(false_negative_states) / denom) if denom > 0 else float('nan')
        return fnr, false_negative_states

    def compute_sat_coverage(self, sat_ids, cells):
        """
        Fraction of total abstraction measure (area/volume) covered by sat states.
        """
        sat = self._normalize_state_ids(sat_ids)
        mins = cells[:, 0::2]
        maxs = cells[:, 1::2]

        cell_measure = np.prod(np.maximum(0.0, maxs - mins), axis=1)
        total_measure = float(np.sum(cell_measure))

        if not sat:
            return 0.0

        sat_idx = np.fromiter(sat, dtype=np.int64)
        sat_measure = float(np.sum(cell_measure[sat_idx]))
        return sat_measure / total_measure

    def compute_dynamics_metrics(self, cells):
        """
        Compute per-cell image metrics used by the reference pipelines.

        Returns dict containing means over source cells:
          - avg_cell_measure
          - avg_image_measure
          - avg_image_over_cell
          - avg_centroid_displacement
          - avg_intersection_over_area
          - avg_intersection_over_union
        """
        n_cells = cells.shape[0]
        mins = cells[:, 0::2]
        maxs = cells[:, 1::2]
        dims = mins.shape[1]

        verts = build_hyperrectangle_vertices(mins, maxs)
        n_corners = verts.shape[1]
        flat_verts = verts.reshape(-1, dims)

        flat_next = np.asarray(self.system.step(flat_verts), dtype=float)
        next_verts = flat_next.reshape(n_cells, n_corners, dims)

        cell_measure = np.prod(np.maximum(0.0, maxs - mins), axis=1)

        img_mins = next_verts.min(axis=1)
        img_maxs = next_verts.max(axis=1)
        image_measure = np.prod(np.maximum(0.0, img_maxs - img_mins), axis=1)

        displacement = np.mean(np.linalg.norm(next_verts - verts, axis=2), axis=1)

        inter_width = np.maximum(0.0, np.minimum(maxs, img_maxs) - np.maximum(mins, img_mins))
        inter_measure = np.prod(inter_width, axis=1)

        image_over_cell = np.full(n_cells, np.nan, dtype=float)
        ioa = np.full(n_cells, np.nan, dtype=float)
        iou = np.full(n_cells, np.nan, dtype=float)

        valid_cell = cell_measure > 0.0
        image_over_cell[valid_cell] = image_measure[valid_cell] / cell_measure[valid_cell]
        ioa[valid_cell] = inter_measure[valid_cell] / cell_measure[valid_cell]

        union_measure = cell_measure + image_measure - inter_measure
        valid_union = union_measure > 0.0
        iou[valid_union] = inter_measure[valid_union] / union_measure[valid_union]

        return {
            'avg_cell_measure': float(np.nanmean(cell_measure)),
            'avg_image_measure': float(np.nanmean(image_measure)),
            'avg_image_over_cell': float(np.nanmean(image_over_cell)),
            'avg_centroid_displacement': float(np.nanmean(displacement)),
            'avg_intersection_over_area': float(np.nanmean(ioa)),
            'avg_intersection_over_union': float(np.nanmean(iou)),
        }

    def evaluate_against_ground_truth(
        self,
        sat_states,
        cells,
        ground_truth_reference,
        gt_reach_regions,
    ):
        """
        Compute the summary metrics used in the reference training scripts.

        Args:
            sat_states: iterable of model-checker satisfying state IDs.
            cells: abstraction cells (N, 2*D).
            ground_truth_reference: dict state_id -> {'goal','fail','unk',...}.
            gt_reach_regions: fixed-grid dict for GT coverage/TNP metrics.

        Returns:
            dict with keys:
              sat_rate, sat_coverage, fnr, true_sat_states, false_negative_states,
              gt_goal_fraction, coverage_proportion, true_negative_fraction.
        """
        n_cells = cells.shape[0]

        checked_sat_states = self._normalize_state_ids(sat_states)
        true_sat_states = {
            int(s) for s, v in ground_truth_reference.items()
            if v == 'goal'
        }

        fnr, false_negative_states = self.false_negative_rate(true_sat_states, checked_sat_states)
        sat_rate = len(checked_sat_states) / n_cells
        sat_coverage = self.compute_sat_coverage(checked_sat_states, cells)

        metrics = {
            'sat_rate': float(sat_rate),
            'sat_coverage': float(sat_coverage),
            'fnr': float(fnr),
            'n_checked_sat_states': int(len(checked_sat_states)),
            'n_true_sat_states': int(len(true_sat_states)),
            'false_negative_states': false_negative_states,
        }

        gt_goal_fraction = (
            sum(1 for v in gt_reach_regions.values() if v == 'goal') / len(gt_reach_regions)
        )
        true_negative_fraction = (
            sum(1 for v in gt_reach_regions.values() if v in ('unk', 'fail')) / len(gt_reach_regions)
        )
        coverage_proportion = (
            sat_coverage / gt_goal_fraction if gt_goal_fraction > 0.0 else float('nan')
        )
        metrics.update({
            'gt_goal_fraction': float(gt_goal_fraction),
            'coverage_proportion': float(coverage_proportion),
            'true_negative_fraction': float(true_negative_fraction),
        })

        return metrics

    def _compute_goal_mask(self, cells):
        raise NotImplementedError

    def _compute_fail_mask(self, cells):
        return np.zeros(cells.shape[0], dtype=bool)

    def _compute_oob_mask(self, cells):
        mins = cells[:, 0::2]
        maxs = cells[:, 1::2]
        domain_min = mins.min(axis=0)
        domain_max = maxs.max(axis=0)

        verts = build_hyperrectangle_vertices(mins, maxs)
        n_cells, n_corners, dims = verts.shape
        flat_verts = verts.reshape(-1, dims)

        flat_next = np.asarray(self.system.step(flat_verts), dtype=float)
        next_verts = flat_next.reshape(n_cells, n_corners, dims)
        oob = (next_verts < domain_min[None, None, :]) | (next_verts > domain_max[None, None, :])
        return np.any(oob, axis=(1, 2))


class SyntheticModelChecker(BaseModelChecker):
    """
    Full model-checking pipeline for synthetic dynamics.

    Ground-truth semantics:
      - 'goal': all corners eventually stay inside the goal ball.
      - 'fail': any corner leaves the domain.
      - 'unk': neither terminal condition reached within max_steps.
    """

    def __init__(
        self,
        system,
        goal_center=(5.0, 5.0),
        goal_radius=2.0,
    ):
        super().__init__(
            system=system,
            include_oob_state=True,
        )
        self.goal_center = np.asarray(goal_center, dtype=float)
        self.goal_radius = float(goal_radius)

    def _compute_goal_mask(self, cells):
        mins = cells[:, 0::2]
        maxs = cells[:, 1::2]
        verts = build_hyperrectangle_vertices(mins, maxs)  # (N, 2^D, D)
        dists = np.linalg.norm(verts - self.goal_center[None, None, :], axis=2)
        return np.all(dists <= self.goal_radius, axis=1)

    def get_gt_reach_regions(self, domain, grid_resolution=100, max_steps=10_000, verbose=False):
        """
        Fixed-grid ground-truth reachability labels over the synthetic domain.
        """
        domain = np.asarray(domain, dtype=float).reshape(-1)

        res_x, res_y = _normalize_grid_resolution(grid_resolution, 2)
        x_min, x_max, y_min, y_max = map(float, domain)
        x_edges = np.linspace(x_min, x_max, res_x)
        y_edges = np.linspace(y_min, y_max, res_y)

        nx = res_x - 1
        ny = res_y - 1
        gt_reach_regions = {}

        count = 0
        total = nx * ny
        for i in range(nx):
            x_lo, x_hi = x_edges[i], x_edges[i + 1]
            for j in range(ny):
                y_lo, y_hi = y_edges[j], y_edges[j + 1]
                verts = np.array(
                    [
                        [x_lo, y_lo],
                        [x_lo, y_hi],
                        [x_hi, y_hi],
                        [x_hi, y_lo],
                    ],
                    dtype=float,
                )

                label = 'unk'
                for _ in range(int(max_steps)):
                    hits_oob = np.any(
                        (verts[:, 0] < x_min)
                        | (verts[:, 0] > x_max)
                        | (verts[:, 1] < y_min)
                        | (verts[:, 1] > y_max)
                    )
                    if hits_oob:
                        label = 'fail'
                        break

                    in_goal = np.all(
                        np.linalg.norm(verts - self.goal_center[None, :], axis=1) <= self.goal_radius
                    )
                    if in_goal:
                        label = 'goal'
                        break

                    verts = np.asarray(self.system.step(verts), dtype=float)

                gt_reach_regions[(i, j)] = label

                if verbose and (count % 10_000 == 0):
                    print(f"    > Processed {count} / {total} regions...")
                count += 1

        return gt_reach_regions

    def check_ground_truth_fast(self, cells, domain, gt_reach_regions):
        """
        Map abstraction cells to GT labels by overlap with fixed-grid GT cells.

        Label aggregation (matching synthetic reference intent):
          fail > unk > goal.
        """
        domain = np.asarray(domain, dtype=float).reshape(-1)

        x_min, x_max, y_min, y_max = map(float, domain)
        domain_min = np.array([x_min, y_min], dtype=float)
        domain_max = np.array([x_max, y_max], dtype=float)

        x_edges, y_edges = _infer_fixed_grid_edges(domain, gt_reach_regions, dims=2)

        mins = cells[:, 0::2]
        maxs = cells[:, 1::2]
        ground_truth_reference = {}

        for src in range(cells.shape[0]):
            cmin = mins[src]
            cmax = maxs[src]

            if np.any(cmin < domain_min) or np.any(cmax > domain_max):
                ground_truth_reference[src] = 'fail'
                continue

            rx = _overlap_index_range(x_edges, cmin[0], cmax[0], upper_side='right')
            ry = _overlap_index_range(y_edges, cmin[1], cmax[1], upper_side='right')

            seen_fail = False
            seen_unk = False
            seen_goal = False

            i_lo, i_hi = rx
            j_lo, j_hi = ry
            for ii in range(i_lo, i_hi + 1):
                for jj in range(j_lo, j_hi + 1):
                    lab = gt_reach_regions[(ii, jj)]
                    if lab == 'fail':
                        seen_fail = True
                        break
                    if lab == 'unk':
                        seen_unk = True
                    elif lab == 'goal':
                        seen_goal = True
                if seen_fail:
                    break

            if seen_fail:
                ground_truth_reference[src] = 'fail'
            elif seen_unk:
                ground_truth_reference[src] = 'unk'
            else:
                ground_truth_reference[src] = 'goal' if seen_goal else 'unk'

        return ground_truth_reference


class MountainCarModelChecker(BaseModelChecker):
    """
    Full model-checking pipeline for closed-loop Mountain Car.

    A cell is labeled as goal iff all of its vertices satisfy:
      position >= goal_position_min
    """

    def __init__(
        self,
        system,
        goal_position_min=0.5,
    ):
        super().__init__(
            system=system,
            include_oob_state=False,
        )
        self.goal_position_min = float(goal_position_min)

    def _compute_goal_mask(self, cells):
        mins = cells[:, 0::2]
        maxs = cells[:, 1::2]
        verts = build_hyperrectangle_vertices(mins, maxs)
        position_coords = verts[:, :, 0]
        return np.all(position_coords >= self.goal_position_min, axis=1)

    def default_ctl_formula(self):
        return 'A (F goal)'

    def get_gt_reach_regions(self, domain, grid_resolution=100, max_steps=10_000, verbose=False):
        """
        Fixed-grid ground-truth labels over Mountain Car state space.

        Labels are {'goal','unk'} to match the reference pipeline.
        """
        domain = np.asarray(domain, dtype=float).reshape(-1)

        res_p, res_v = _normalize_grid_resolution(grid_resolution, 2)
        p_min, p_max, v_min, v_max = map(float, domain)
        p_edges = np.linspace(p_min, p_max, res_p)
        v_edges = np.linspace(v_min, v_max, res_v)

        np_cells = res_p - 1
        nv_cells = res_v - 1
        gt_reach_regions = {}

        count = 0
        total = np_cells * nv_cells
        for i in range(np_cells):
            p_lo, p_hi = p_edges[i], p_edges[i + 1]
            for j in range(nv_cells):
                v_lo, v_hi = v_edges[j], v_edges[j + 1]
                verts = np.array(
                    [
                        [p_lo, v_lo],
                        [p_lo, v_hi],
                        [p_hi, v_hi],
                        [p_hi, v_lo],
                    ],
                    dtype=float,
                )

                label = 'unk'
                for _ in range(int(max_steps)):
                    in_goal = np.all(verts[:, 0] >= self.goal_position_min)
                    if in_goal:
                        label = 'goal'
                        break

                    verts = np.asarray(self.system.step(verts), dtype=float)

                gt_reach_regions[(i, j)] = label

                if verbose and (count % 10_000 == 0):
                    print(f"    > Processed {count} / {total} regions...")
                count += 1

        return gt_reach_regions

    def check_ground_truth_fast(self, cells, domain, gt_reach_regions):
        """
        Map abstraction cells to GT labels by overlap with fixed-grid GT cells.

        Label aggregation:
          - 'goal' only if overlapping GT cells are all goal.
          - otherwise 'unk' (or 'fail' if source lies outside domain).
        """
        domain = np.asarray(domain, dtype=float).reshape(-1)

        p_min, p_max, v_min, v_max = map(float, domain)
        domain_min = np.array([p_min, v_min], dtype=float)
        domain_max = np.array([p_max, v_max], dtype=float)

        p_edges, v_edges = _infer_fixed_grid_edges(domain, gt_reach_regions, dims=2)

        mins = cells[:, 0::2]
        maxs = cells[:, 1::2]
        ground_truth_reference = {}

        for src in range(cells.shape[0]):
            cmin = mins[src]
            cmax = maxs[src]

            if np.any(cmin < domain_min) or np.any(cmax > domain_max):
                ground_truth_reference[src] = 'fail'
                continue

            rp = _overlap_index_range(p_edges, cmin[0], cmax[0], upper_side='right')
            rv = _overlap_index_range(v_edges, cmin[1], cmax[1], upper_side='right')

            all_goal = True
            i_lo, i_hi = rp
            j_lo, j_hi = rv
            for ii in range(i_lo, i_hi + 1):
                for jj in range(j_lo, j_hi + 1):
                    if gt_reach_regions[(ii, jj)] != 'goal':
                        all_goal = False
                        break
                if not all_goal:
                    break

            ground_truth_reference[src] = 'goal' if all_goal else 'unk'

        return ground_truth_reference


class UnicycleModelChecker(BaseModelChecker):
    """
    Full model-checking pipeline for closed-loop unicycle dynamics.

    Labeling semantics (matching abstraction-training/uni_model_checking_tools.py):
      - goal: all cell vertices are inside goal disc in (x, y)
      - fail: any cell vertex is inside any obstacle disc in (x, y)
      - safe: otherwise
    """

    def __init__(
        self,
        system,
        goal_center=(40.0, 20.0),
        goal_radius=8.0,
        obstacle_centers=((25.0, 25.0),),
        obstacle_radii=(5.0,),
    ):
        super().__init__(
            system=system,
            include_oob_state=True,
        )
        self.goal_center = np.asarray(goal_center, dtype=float)
        self.goal_radius = float(goal_radius)
        self.obstacle_centers = np.atleast_2d(np.asarray(obstacle_centers, dtype=float))
        self.obstacle_radii = np.atleast_1d(np.asarray(obstacle_radii, dtype=float))

    def _goal_all_vertices(self, verts):
        xy = verts[:, :2]
        d = np.linalg.norm(xy - self.goal_center[None, :], axis=1)
        return bool(np.all(d <= self.goal_radius))

    def _obstacle_any_vertex(self, verts):
        xy = verts[:, :2]
        for center, radius in zip(self.obstacle_centers, self.obstacle_radii):
            d = np.linalg.norm(xy - center[None, :], axis=1)
            if np.any(d <= radius):
                return True
        return False

    def _compute_goal_mask(self, cells):
        mins = cells[:, 0::2]
        maxs = cells[:, 1::2]
        verts = build_hyperrectangle_vertices(mins, maxs)  # (N, 2^D, D)
        xy = verts[:, :, :2]
        dists = np.linalg.norm(xy - self.goal_center[None, None, :], axis=2)
        return np.all(dists <= self.goal_radius, axis=1)

    def _compute_fail_mask(self, cells):
        mins = cells[:, 0::2]
        maxs = cells[:, 1::2]
        verts = build_hyperrectangle_vertices(mins, maxs)
        xy = verts[:, :, :2]

        fail = np.zeros(cells.shape[0], dtype=bool)
        for center, radius in zip(self.obstacle_centers, self.obstacle_radii):
            dists = np.linalg.norm(xy - center[None, None, :], axis=2)
            fail |= np.any(dists <= radius, axis=1)
        return fail

    def _compute_oob_mask(self, cells):
        """
        Match unicycle reference behavior: OOB is defined on (x, y) only.
        """
        mins = cells[:, 0::2]
        maxs = cells[:, 1::2]

        domain_min = mins.min(axis=0)
        domain_max = maxs.max(axis=0)
        x_min, x_max = domain_min[0], domain_max[0]
        y_min, y_max = domain_min[1], domain_max[1]

        verts = build_hyperrectangle_vertices(mins, maxs)
        n_cells, n_corners, dims = verts.shape
        flat_verts = verts.reshape(-1, dims)

        flat_next = np.asarray(self.system.step(flat_verts), dtype=float)
        next_verts = flat_next.reshape(n_cells, n_corners, dims)
        x_next = next_verts[:, :, 0]
        y_next = next_verts[:, :, 1]
        return np.any((x_next < x_min) | (x_next > x_max) | (y_next < y_min) | (y_next > y_max), axis=1)

    def get_gt_reach_regions(self, domain, grid_resolution=100, max_steps=10_000, verbose=False):
        """
        Fixed-grid ground-truth labels over unicycle state space.

        Labels are {'goal','fail','unk'}.
        """
        domain = np.asarray(domain, dtype=float).reshape(-1)

        res_x, res_y, res_t = _normalize_grid_resolution(grid_resolution, 3)
        x_min, x_max, y_min, y_max, t_min, t_max = map(float, domain)
        x_edges = np.linspace(x_min, x_max, res_x)
        y_edges = np.linspace(y_min, y_max, res_y)
        t_edges = np.linspace(t_min, t_max, res_t)

        nx = res_x - 1
        ny = res_y - 1
        nt = res_t - 1
        gt_reach_regions = {}

        count = 0
        total = nx * ny * nt
        for i in range(nx):
            x_lo, x_hi = x_edges[i], x_edges[i + 1]
            for j in range(ny):
                y_lo, y_hi = y_edges[j], y_edges[j + 1]
                for k in range(nt):
                    t_lo, t_hi = t_edges[k], t_edges[k + 1]

                    verts = np.array(
                        [
                            [x_lo, y_lo, t_lo],
                            [x_lo, y_hi, t_lo],
                            [x_hi, y_hi, t_lo],
                            [x_hi, y_lo, t_lo],
                            [x_lo, y_lo, t_hi],
                            [x_lo, y_hi, t_hi],
                            [x_hi, y_hi, t_hi],
                            [x_hi, y_lo, t_hi],
                        ],
                        dtype=float,
                    )

                    label = 'unk'
                    for _ in range(int(max_steps)):
                        if self._obstacle_any_vertex(verts):
                            label = 'fail'
                            break

                        if self._goal_all_vertices(verts):
                            label = 'goal'
                            break

                        oob_xy = np.any(
                            (verts[:, 0] < x_min)
                            | (verts[:, 0] > x_max)
                            | (verts[:, 1] < y_min)
                            | (verts[:, 1] > y_max)
                        )
                        if oob_xy:
                            label = 'fail'
                            break

                        verts = np.asarray(self.system.step(verts), dtype=float)

                    gt_reach_regions[(i, j, k)] = label

                    if verbose and (count % 10_000 == 0):
                        print(f"    > Processed {count} / {total} regions...")
                    count += 1

        return gt_reach_regions

    def check_ground_truth_fast(
        self,
        cells,
        domain,
        gt_reach_regions,
    ):
        """
        Fast GT mapping for unicycle abstraction cells.

        Goal iff all overlapping fixed GT cells are goal, fail otherwise.
        """
        domain = np.asarray(domain, dtype=float).reshape(-1)

        x_min, x_max, y_min, y_max, t_min, t_max = map(float, domain)
        domain_min = np.array([x_min, y_min, t_min], dtype=float)
        domain_max = np.array([x_max, y_max, t_max], dtype=float)

        x_edges, y_edges, t_edges = _infer_fixed_grid_edges(domain, gt_reach_regions, dims=3)

        mins = cells[:, 0::2]
        maxs = cells[:, 1::2]
        ground_truth_reference = {}

        for src in range(cells.shape[0]):
            cmin = mins[src]
            cmax = maxs[src]

            if np.any(cmin < domain_min) or np.any(cmax > domain_max):
                ground_truth_reference[src] = 'fail'
                continue

            rx = _overlap_index_range(x_edges, cmin[0], cmax[0])
            ry = _overlap_index_range(y_edges, cmin[1], cmax[1])
            rt = _overlap_index_range(t_edges, cmin[2], cmax[2])

            i_lo, i_hi = rx
            j_lo, j_hi = ry
            k_lo, k_hi = rt

            all_goal = True
            for ii in range(i_lo, i_hi + 1):
                for jj in range(j_lo, j_hi + 1):
                    for kk in range(k_lo, k_hi + 1):
                        if gt_reach_regions[(ii, jj, kk)] != 'goal':
                            all_goal = False
                            break
                    if not all_goal:
                        break
                if not all_goal:
                    break

            ground_truth_reference[src] = 'goal' if all_goal else 'fail'

        return ground_truth_reference
