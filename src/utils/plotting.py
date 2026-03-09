import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Patch

def plot_grid_abstraction(
    cells,
    x1_space,
    x2_space,
    output_path='out/grid_abstraction.png',
    title=None,
    show=False,
    color='blue',
    alpha=0.2,
    linewidth=0.5,
    system=None,
    subsample=10,
    print_saved=False
):
    """
    Plots the grid abstraction and optional vector field.

    Args:
        cells: (N, 4) numpy array of [x1_min, x1_max, x2_min, x2_max]
        x1_space: tuple (min, max) for x1 axis
        x2_space: tuple (min, max) for x2 axis
        output_path: path to save the figure
        title: title of the plot
        show: whether to display the plot (blocking)
        color: color of the grid lines
        alpha: transparency of grid fill/lines
        linewidth: width of grid lines
        system: optional system object with a step() method for vector field
        subsample: step size for subsampling the grid points for vector plotting
        print_saved: whether to print save-path message after writing figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    x1_lines = np.unique(np.concatenate([cells[:, 0], cells[:, 1]]))
    x2_lines = np.unique(np.concatenate([cells[:, 2], cells[:, 3]]))

    for x in x1_lines:
        ax.axvline(x, color=color, linewidth=linewidth, alpha=alpha)

    for y in x2_lines:
        ax.axhline(y, color=color, linewidth=linewidth, alpha=alpha)

    if system is not None:
        gx = np.linspace(x1_space[0], x1_space[1], 40)
        gy = np.linspace(x2_space[0], x2_space[1], 40)
        GX, GY = np.meshgrid(gx, gy)

        flat_states = np.column_stack([GX.ravel(), GY.ravel()])
        next_states = system.step(flat_states)
        disps = next_states - flat_states

        U = disps[:, 0] * 0.3
        V = disps[:, 1] * 0.3

        ax.quiver(GX.ravel(), GY.ravel(), U, V, color='red', alpha=0.5, angles='xy', scale_units='xy', scale=1, pivot='tail')

    ax.set_xlim(x1_space)
    ax.set_ylim(x2_space)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        if print_saved:
            print(f"Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def _grid_2d_axes(cells):
    """
    Return 2D grid lines and cell counts for canonical generate_grid ordering.
    """
    x_lines = np.unique(np.concatenate((cells[:, 0], cells[:, 1])))
    y_lines = np.unique(np.concatenate((cells[:, 2], cells[:, 3])))
    nx = x_lines.size - 1
    ny = y_lines.size - 1
    return x_lines, y_lines, nx, ny


def _cell_values_to_grid(cells, values):
    """
    Convert per-cell values (N,) into a grid array shaped (ny, nx) for pcolormesh.
    """
    values = np.asarray(values, dtype=float)
    x_lines, y_lines, nx, ny = _grid_2d_axes(cells)
    grid = values.reshape(nx, ny).T
    return x_lines, y_lines, grid


def _transition_metric_values(transition_map, metric):
    """
    Compute per-state scalar metrics from a transition map.
    """
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


def plot_transition_heatmap(
    cells,
    transition_map,
    metric='out_degree',
    output_path='out/transition_heatmap.png',
    title=None,
    show=False,
    cmap='viridis',
    vmin=None,
    vmax=None,
    print_saved=False,
):
    """
    Plot per-cell transition statistics as a 2D heatmap.

    Supported metrics:
        out_degree, in_degree, self_loop, self_loop_only, nonempty, out_minus_in
    """
    vals, label = _transition_metric_values(transition_map, metric)
    x_lines, y_lines, grid = _cell_values_to_grid(cells, vals)

    fig, ax = plt.subplots(figsize=(8, 8))
    mesh = ax.pcolormesh(x_lines, y_lines, grid, cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(label)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_aspect('equal')
    if title is None:
        title = f'Transition Heatmap: {metric}'
    ax.set_title(title)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        if print_saved:
            print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_transition_comparison_heatmap(
    cells,
    transition_map_a,
    transition_map_b,
    metric='state_jaccard',
    output_path='out/transition_compare_heatmap.png',
    title=None,
    show=False,
    cmap='magma',
    vmin=None,
    vmax=None,
    print_saved=False,
):
    """
    Plot per-cell disagreement/agreement between two transition maps.

    Supported metrics:
        state_jaccard, symmetric_diff_count, edge_count_delta, exact_match
    """
    n = len(cells)
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

    x_lines, y_lines, grid = _cell_values_to_grid(cells, vals)

    if metric == 'state_jaccard' and vmin is None and vmax is None:
        vmin, vmax = 0.0, 1.0
    if metric == 'exact_match' and vmin is None and vmax is None:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    mesh = ax.pcolormesh(x_lines, y_lines, grid, cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(metric)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_aspect('equal')
    if title is None:
        title = f'Transition Comparison: {metric}'
    ax.set_title(title)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        if print_saved:
            print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_false_negative_map(
    cells,
    sat_states,
    false_negative_states,
    output_path='out/false_negative_map.png',
    title='Model Checking Results',
    show=False,
    alpha=0.6,
    edge_alpha=0.15,
    edge_linewidth=0.35,
    x1_space=None,
    x2_space=None,
    goal_center=None,
    goal_radius=None,
    show_legend=True,
    print_saved=False,
):
    """
    Plot a 2D TP/FN/TN map with the same style semantics as abstraction-training.

    Coloring convention:
        - True positives (sat): green
        - False negatives (safe but unsat): light blue
        - True negatives (unsat): red
    """
    n_cells = cells.shape[0]
    sat_states = {int(s) for s in sat_states}
    false_negative_states = {int(s) for s in false_negative_states}

    tp_color = (0.2, 0.7, 0.2, alpha)
    fn_color = (0.55, 0.80, 1.0, alpha)
    tn_color = (0.85, 0.2, 0.2, alpha)

    polys = []
    facecolors = []
    edgecolors = []
    linewidths = []

    for sid, cell in enumerate(cells):
        x1_lo, x1_hi, x2_lo, x2_hi = cell
        polys.append(
            np.array(
                [
                    [x1_lo, x2_lo],
                    [x1_lo, x2_hi],
                    [x1_hi, x2_hi],
                    [x1_hi, x2_lo],
                ],
                dtype=float,
            )
        )

        # Keep reference precedence: sat overrides false-negative if both appear.
        if sid in sat_states:
            facecolors.append(tp_color)
        elif sid in false_negative_states:
            facecolors.append(fn_color)
        else:
            facecolors.append(tn_color)

        edgecolors.append((0.0, 0.0, 0.0, edge_alpha))
        linewidths.append(edge_linewidth)

    fig, ax = plt.subplots(figsize=(8, 8))
    coll = PolyCollection(
        polys,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=linewidths,
        closed=True,
    )
    ax.add_collection(coll)

    if goal_center is not None and goal_radius is not None:
        goal_center = np.asarray(goal_center, dtype=float).reshape(2,)
        circle = Circle(
            (float(goal_center[0]), float(goal_center[1])),
            float(goal_radius),
            fill=False,
            color='k',
            linewidth=2.0,
        )
        ax.add_patch(circle)

    if x1_space is not None and x2_space is not None:
        x1_min, x1_max = map(float, x1_space)
        x2_min, x2_max = map(float, x2_space)
    else:
        x1_min = float(cells[:, 0].min())
        x1_max = float(cells[:, 1].max())
        x2_min = float(cells[:, 2].min())
        x2_max = float(cells[:, 3].max())

    ax.plot(
        [x1_min, x1_min, x1_max, x1_max, x1_min],
        [x2_min, x2_max, x2_max, x2_min, x2_min],
        color='k',
        linewidth=1.5,
    )
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

    if show_legend:
        # Reserve room on the right and place legend outside axes to avoid map overlap.
        fig.subplots_adjust(right=0.80)
        ax.legend(
            handles=[
                Patch(facecolor=tp_color, edgecolor='none', label='True Positive (sat)'),
                Patch(facecolor=fn_color, edgecolor='none', label='False Negative'),
                Patch(facecolor=tn_color, edgecolor='none', label='True Negative (unsat)'),
            ],
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            framealpha=0.9,
        )

    ax.set_title(title)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        if print_saved:
            print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)
