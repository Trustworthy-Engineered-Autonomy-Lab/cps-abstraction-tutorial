import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def _grid_3d_axes(cells):
    """
    Return 3D grid lines and cell counts for canonical generate_grid ordering.
    """
    x_lines = np.unique(np.concatenate((cells[:, 0], cells[:, 1])))
    y_lines = np.unique(np.concatenate((cells[:, 2], cells[:, 3])))
    z_lines = np.unique(np.concatenate((cells[:, 4], cells[:, 5])))
    nx = x_lines.size - 1
    ny = y_lines.size - 1
    nz = z_lines.size - 1
    return x_lines, y_lines, z_lines, nx, ny, nz


def _cell_values_to_grid_3d(cells, values):
    """
    Convert per-cell values (N,) into a grid array shaped (nz, ny, nx).
    """
    values = np.asarray(values, dtype=float)
    x_lines, y_lines, z_lines, nx, ny, nz = _grid_3d_axes(cells)
    grid = values.reshape(nx, ny, nz).transpose(2, 1, 0)
    return x_lines, y_lines, z_lines, grid


def _transition_metric_values(transition_map, metric):
    n = len(transition_map)

    out_degree = np.array([len(s) for s in transition_map], dtype=float)
    in_degree = np.zeros(n, dtype=float)
    self_loop = np.zeros(n, dtype=float)
    for i, succ in enumerate(transition_map):
        if i in succ:
            self_loop[i] = 1.0
        for j in succ:
            in_degree[j] += 1.0

    if metric == 'out_degree':
        return out_degree, 'Out-degree'
    if metric == 'in_degree':
        return in_degree, 'In-degree'
    if metric == 'self_loop':
        return self_loop, 'Self-loop (0/1)'
    if metric == 'self_loop_only':
        vals = np.array([1.0 if s == {i} else 0.0 for i, s in enumerate(transition_map)], dtype=float)
        return vals, 'Self-loop only (0/1)'
    if metric == 'nonempty':
        return (out_degree > 0).astype(float), 'Has successor (0/1)'
    if metric == 'out_minus_in':
        return out_degree - in_degree, 'Out-degree - In-degree'

    raise ValueError(
        "Unknown metric. Use one of: "
        "out_degree, in_degree, self_loop, self_loop_only, nonempty, out_minus_in"
    )


def _comparison_metric_values(transition_map_a, transition_map_b, metric):
    n = len(transition_map_a)
    vals = np.zeros(n, dtype=float)
    for i, (a, b) in enumerate(zip(transition_map_a, transition_map_b)):
        if metric == 'state_jaccard':
            u = len(a | b)
            vals[i] = (len(a & b) / u) if u > 0 else 1.0
        elif metric == 'symmetric_diff_count':
            vals[i] = float(len(a ^ b))
        elif metric == 'edge_count_delta':
            vals[i] = float(len(a) - len(b))
        elif metric == 'exact_match':
            vals[i] = 1.0 if a == b else 0.0
        else:
            raise ValueError(
                "Unknown metric. Use one of: "
                "state_jaccard, symmetric_diff_count, edge_count_delta, exact_match"
            )
    return vals, metric


def _select_slice_indices(nz, theta_indices=None, max_slices=6):
    if theta_indices is None:
        n = int(max(1, min(max_slices, nz)))
        idx = np.linspace(0, nz - 1, num=n, dtype=int)
    else:
        idx = np.asarray(theta_indices, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < nz)]
    idx = np.unique(idx)
    if idx.size == 0:
        raise ValueError("no valid theta slice indices selected")
    return idx


def _reduce_3d_grid(grid_3d, reduce):
    if reduce == 'mean':
        return np.nanmean(grid_3d, axis=0), 'mean'
    if reduce == 'max':
        return np.nanmax(grid_3d, axis=0), 'max'
    if reduce == 'min':
        return np.nanmin(grid_3d, axis=0), 'min'
    if reduce == 'sum':
        return np.nansum(grid_3d, axis=0), 'sum'
    raise ValueError("Unknown reduce. Use one of: mean, max, min, sum")


def plot_transition_theta_slices(
    cells,
    transition_map,
    metric='out_degree',
    theta_indices=None,
    max_slices=6,
    output_path='out/transition_theta_slices.png',
    title=None,
    show=False,
    cmap='viridis',
    vmin=None,
    vmax=None,
    print_saved=False,
):
    """
    Plot x-y heatmaps for selected theta slices of a 3D transition map.
    """
    vals, label = _transition_metric_values(transition_map, metric)
    x_lines, y_lines, z_lines, grid_3d = _cell_values_to_grid_3d(cells, vals)
    nz = grid_3d.shape[0]

    theta_idx = _select_slice_indices(nz, theta_indices=theta_indices, max_slices=max_slices)
    nplots = theta_idx.size
    ncols = min(3, nplots)
    nrows = int(np.ceil(nplots / ncols))

    if vmin is None:
        vmin = np.nanmin(grid_3d[theta_idx])
    if vmax is None:
        vmax = np.nanmax(grid_3d[theta_idx])

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.5 * nrows), squeeze=False)
    mappable = None

    for ax_i, k in enumerate(theta_idx):
        r = ax_i // ncols
        c = ax_i % ncols
        ax = axes[r, c]
        mappable = ax.pcolormesh(
            x_lines,
            y_lines,
            grid_3d[k],
            cmap=cmap,
            shading='flat',
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"Theta slice {int(k)} [{z_lines[k]:.3f}, {z_lines[k + 1]:.3f}]")

    for ax_i in range(nplots, nrows * ncols):
        r = ax_i // ncols
        c = ax_i % ncols
        axes[r, c].axis('off')

    # Reserve right margin and place colorbar in its own axis to avoid overlap.
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.90, wspace=0.25, hspace=0.30)
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.74])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label)

    if title is None:
        title = f'{metric}: Theta Slices'
    fig.suptitle(title)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        if print_saved:
            print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_transition_theta_projection(
    cells,
    transition_map,
    metric='out_degree',
    reduce='mean',
    output_path='out/transition_theta_projection.png',
    title=None,
    show=False,
    cmap='viridis',
    vmin=None,
    vmax=None,
    print_saved=False,
):
    """
    Plot x-y heatmap after reducing a 3D metric over theta.
    """
    vals, label = _transition_metric_values(transition_map, metric)
    x_lines, y_lines, _, grid_3d = _cell_values_to_grid_3d(cells, vals)
    grid_2d, reduce_label = _reduce_3d_grid(grid_3d, reduce=reduce)

    fig, ax = plt.subplots(figsize=(8, 8))
    mesh = ax.pcolormesh(x_lines, y_lines, grid_2d, cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(f'{label} ({reduce_label} over theta)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    if title is None:
        title = f'{metric} ({reduce_label} over theta)'
    ax.set_title(title)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        if print_saved:
            print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_transition_comparison_theta_slices(
    cells,
    transition_map_a,
    transition_map_b,
    metric='state_jaccard',
    theta_indices=None,
    max_slices=6,
    output_path='out/transition_compare_theta_slices.png',
    title=None,
    show=False,
    cmap='magma',
    vmin=None,
    vmax=None,
    print_saved=False,
):
    """
    Plot x-y disagreement/agreement heatmaps for selected theta slices.
    """
    vals, label = _comparison_metric_values(transition_map_a, transition_map_b, metric)
    x_lines, y_lines, z_lines, grid_3d = _cell_values_to_grid_3d(cells, vals)
    nz = grid_3d.shape[0]

    theta_idx = _select_slice_indices(nz, theta_indices=theta_indices, max_slices=max_slices)
    nplots = theta_idx.size
    ncols = min(3, nplots)
    nrows = int(np.ceil(nplots / ncols))

    if metric in ('state_jaccard', 'exact_match') and vmin is None and vmax is None:
        vmin, vmax = 0.0, 1.0
    if vmin is None:
        vmin = np.nanmin(grid_3d[theta_idx])
    if vmax is None:
        vmax = np.nanmax(grid_3d[theta_idx])

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.5 * nrows), squeeze=False)
    mappable = None

    for ax_i, k in enumerate(theta_idx):
        r = ax_i // ncols
        c = ax_i % ncols
        ax = axes[r, c]
        mappable = ax.pcolormesh(
            x_lines,
            y_lines,
            grid_3d[k],
            cmap=cmap,
            shading='flat',
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"Theta slice {int(k)} [{z_lines[k]:.3f}, {z_lines[k + 1]:.3f}]")

    for ax_i in range(nplots, nrows * ncols):
        r = ax_i // ncols
        c = ax_i % ncols
        axes[r, c].axis('off')

    # Reserve right margin and place colorbar in its own axis to avoid overlap.
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.90, wspace=0.25, hspace=0.30)
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.74])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label)

    if title is None:
        title = f'Comparison ({metric}): Theta Slices'
    fig.suptitle(title)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        if print_saved:
            print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_transition_comparison_theta_projection(
    cells,
    transition_map_a,
    transition_map_b,
    metric='state_jaccard',
    reduce='mean',
    output_path='out/transition_compare_theta_projection.png',
    title=None,
    show=False,
    cmap='magma',
    vmin=None,
    vmax=None,
    print_saved=False,
):
    """
    Plot x-y disagreement/agreement heatmap after reducing over theta.
    """
    vals, label = _comparison_metric_values(transition_map_a, transition_map_b, metric)
    x_lines, y_lines, _, grid_3d = _cell_values_to_grid_3d(cells, vals)
    grid_2d, reduce_label = _reduce_3d_grid(grid_3d, reduce=reduce)

    if metric in ('state_jaccard', 'exact_match') and vmin is None and vmax is None:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    mesh = ax.pcolormesh(x_lines, y_lines, grid_2d, cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(f'{label} ({reduce_label} over theta)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    if title is None:
        title = f'Comparison {metric} ({reduce_label} over theta)'
    ax.set_title(title)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        if print_saved:
            print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def _state_class_values(n_cells, sat_states, false_negative_states):
    """
    Build per-state class codes for FN map plotting.

    Codes:
        0 -> TN (unsat)
        1 -> FN (safe but unsat)
        2 -> TP (sat)
    """
    sat = {int(s) for s in sat_states}
    fn = {int(s) for s in false_negative_states}

    vals = np.zeros(n_cells, dtype=float)
    if fn:
        vals[list(fn)] = 1.0
    if sat:
        # Keep same precedence as 2D map/reference: sat overrides fn if both present.
        vals[list(sat)] = 2.0
    return vals


def plot_false_negative_theta_slices(
    cells,
    sat_states,
    false_negative_states,
    theta_indices=None,
    max_slices=6,
    output_path='out/false_negative_theta_slices.png',
    title='False-Negative Map: Theta Slices',
    show=False,
    alpha=0.70,
    print_saved=False,
):
    """
    Plot TP/FN/TN categorical maps over selected theta slices.
    """
    n_cells = len(cells)
    vals = _state_class_values(n_cells, sat_states, false_negative_states)
    x_lines, y_lines, z_lines, grid_3d = _cell_values_to_grid_3d(cells, vals)
    nz = grid_3d.shape[0]

    theta_idx = _select_slice_indices(nz, theta_indices=theta_indices, max_slices=max_slices)
    nplots = theta_idx.size
    ncols = min(3, nplots)
    nrows = int(np.ceil(nplots / ncols))

    tp_color = (0.2, 0.7, 0.2, alpha)
    fn_color = (0.55, 0.80, 1.0, alpha)
    tn_color = (0.85, 0.2, 0.2, alpha)
    cmap = ListedColormap([tn_color, fn_color, tp_color])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.5 * nrows), squeeze=False)

    for ax_i, k in enumerate(theta_idx):
        r = ax_i // ncols
        c = ax_i % ncols
        ax = axes[r, c]
        ax.pcolormesh(
            x_lines,
            y_lines,
            grid_3d[k],
            cmap=cmap,
            norm=norm,
            shading='flat',
        )
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"Theta slice {int(k)} [{z_lines[k]:.3f}, {z_lines[k + 1]:.3f}]")

    for ax_i in range(nplots, nrows * ncols):
        r = ax_i // ncols
        c = ax_i % ncols
        axes[r, c].axis('off')

    fig.subplots_adjust(left=0.08, right=0.82, bottom=0.10, top=0.90, wspace=0.25, hspace=0.30)
    fig.suptitle(title)
    fig.legend(
        handles=[
            Patch(facecolor=tp_color, edgecolor='none', label='True Positive (sat)'),
            Patch(facecolor=fn_color, edgecolor='none', label='False Negative'),
            Patch(facecolor=tn_color, edgecolor='none', label='True Negative (unsat)'),
        ],
        loc='center left',
        bbox_to_anchor=(0.84, 0.5),
        ncol=1,
        borderaxespad=0.0,
        framealpha=0.9,
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        if print_saved:
            print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_false_negative_theta_projection(
    cells,
    sat_states,
    false_negative_states,
    metric='fn_fraction',
    output_path='out/false_negative_theta_projection.png',
    title=None,
    show=False,
    cmap=None,
    vmin=0.0,
    vmax=1.0,
    print_saved=False,
):
    """
    Plot a theta-projected FN diagnostic map.

    Supported metrics:
        fn_fraction, tp_fraction, tn_fraction
    """
    n_cells = len(cells)
    vals = _state_class_values(n_cells, sat_states, false_negative_states)
    x_lines, y_lines, _, grid_3d = _cell_values_to_grid_3d(cells, vals)

    if metric == 'fn_fraction':
        grid_2d = np.mean(grid_3d == 1.0, axis=0)
        label = 'FN fraction over theta'
        cmap = cmap or 'Blues'
    elif metric == 'tp_fraction':
        grid_2d = np.mean(grid_3d == 2.0, axis=0)
        label = 'TP fraction over theta'
        cmap = cmap or 'Greens'
    elif metric == 'tn_fraction':
        grid_2d = np.mean(grid_3d == 0.0, axis=0)
        label = 'TN fraction over theta'
        cmap = cmap or 'Reds'
    else:
        raise ValueError("Unknown metric. Use one of: fn_fraction, tp_fraction, tn_fraction")

    fig, ax = plt.subplots(figsize=(8, 8))
    mesh = ax.pcolormesh(
        x_lines,
        y_lines,
        grid_2d,
        cmap=cmap,
        shading='flat',
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(label)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    if title is None:
        title = f'False-Negative Projection: {metric}'
    ax.set_title(title)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        if print_saved:
            print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)
