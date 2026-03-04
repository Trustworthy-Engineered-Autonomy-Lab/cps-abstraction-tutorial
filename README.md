# A Pragmatic Guide to Building Conservative Discrete Abstractions of Cyber-Physical Systems

This repository provides the full code for building conservative discrete abstractions of the systems featured in the paper "A Pragmatic Guide to Building Conservative Discrete Abstractions of Cyber-Physical Systems."
The full, extended version of the paper is [available on arXiv](https://arxiv.org/).

## Repository Purpose

The repository implements end-to-end pipelines for three benchmark systems (`synthetic`, `mountain_car`, and `unicycle`). Each pipeline performs the same sequence of tasks:

1. Construct a uniform grid over the continuous state space.
2. Compute one-step transition relations using multiple successor-generation methods. (Optionally can run `_with_self_loop_removal` variants to demonstrate verified self-loop removal)
3. Build a Kripke structure and run CTL model checking.
4. Compare abstraction results against a fixed-grid ground-truth estimate.
5. Produce diagnostic figures and runtime summaries.

## Requirements and Installation

### Runtime Requirements

- Python 3.x
- `pip`
- Python dependencies listed in `requirements.txt`

### Python Dependencies

- `numpy`
- `matplotlib`
- `torch`
- `scipy`
- `pyModelChecking`
- `colorama`
- `tabulate`

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipelines

Execute any pipeline directly:

```bash
python abstract_synthetic.py
python abstract_mountain_car.py
python abstract_unicycle.py
python abstract_synthetic_with_self_loop_removal.py
python abstract_mountain_car_with_self_loop_removal.py
python abstract_unicycle_with_self_loop_removal.py
```

Each script defines an `ARGS` configuration dictionary in `__main__` for:

- grid resolution and domain parameters
- enabled transition methods (`run_aabb`, `run_poly`, `run_sample`)
- sampling parameters (`sample_n`, `sample_seed`)
- Kripke/model-checking controls
- ground-truth evaluation controls
- plotting controls

Additional `ARGS` in the _with_self_loop_removal variants:

- max steps for reachable-set propagation certificate (`[method]_self_loop_max_steps`)
- number of samples for sample-based certificate (`[method]_sample_exit_n`)
- max steps per sample (`[method]_sample_exit_max_steps`)

Artifacts are written to `out/<system>/`, and ground-truth caches are stored in `cache/`.

## Successor (Transition) Methods

The transition builders are implemented in `helpers/partitioning.py`. All methods return a transition map with the same interface:

- `transition_map[i]` is a `set[int]` containing the one-step successor cell indices of source cell `i`

### Common Grid and Geometry Preparation

- `generate_grid(...)` constructs a uniform rectilinear grid and indexing metadata (grid lines, dimensions, flat-index strides).
- The geometric methods (`AABB` and `POLY`) propagate all source-cell vertices through `system.step(...)` in vectorized form.
- When `periodic_theta=True`, dimension index `2` is treated as periodic (used by the unicycle pipeline).

### `AABB` (`compute_transitions_AABB`)

`AABB` computes successors by over-approximating the image of each source cell with an axis-aligned bounding box formed from propagated source-cell vertices, then adding every destination cell overlapped by that box.

### `POLY` (`compute_transitions_poly`)

`POLY` begins from the `AABB` candidate set and prunes it by testing intersection between each candidate destination cell and the convex hull of propagated source-cell vertices.

### `SAMPLE` (`compute_transitions_sample`)

`SAMPLE` estimates successors empirically by drawing batches of states over the domain, binning sampled source and stepped destination states into grid cells, and recording observed `(source, destination)` pairs. Sampling terminates when a Good-Turing style missing-mass upper bound falls below `beta` (with confidence parameter `delta`).

## Project Structure

The repository is organized into pipeline entry points, reusable helper modules, system definitions, and generated artifacts.

### Pipeline Entry Points

| Path | Role |
| --- | --- |
| `abstract_synthetic.py` | End-to-end abstraction, model-checking, and evaluation pipeline for the synthetic system. |
| `abstract_mountain_car.py` | End-to-end pipeline for the mountain car system. |
| `abstract_unicycle.py` | End-to-end pipeline for the unicycle system (`x`, `y`, `theta`). |
| `abstract_synthetic_self_with_loop_removal.py` | End-to-end abstraction, model-checking, and evaluation pipeline for the synthetic system with self loop removal demonstration. |
| `abstract_mountain_car_with_loop_removal.py` | End-to-end pipeline for the mountain car system with self loop removal demonstration. |
| `abstract_unicycle_with_loop_removal.py` | End-to-end pipeline for the unicycle system (`x`, `y`, `theta`) with self loop removal demonstration. |

### Core Helper Modules (`helpers/`)

| Path | Role |
| --- | --- |
| `helpers/partitioning.py` | Uniform grid construction and transition-map generation (`AABB`, `POLY`, `SAMPLE`). |
| `helpers/math_utils.py` | Geometric and convex-hull utilities used by transition refinement and intersection tests. |
| `helpers/model_checking_tools.py` | Kripke-structure construction, system-specific labeling, and CTL model-checking utilities. |
| `helpers/ground_truth_cache.py` | Cache key/path construction and persistence for ground-truth evaluations. |
| `helpers/log_utils.py` | Runtime measurement, stage logging, and formatted reporting utilities. |
| `helpers/plotting.py` | Plotting utilities for 2D systems (synthetic and mountain car). |
| `helpers/plotting_3d.py` | Plotting utilities for the unicycle state space and theta projections/slices. |
| `helpers/self_loop.py` | Self-loop removal functions for the synthetic and mountain car cases. |
| `helpers/self_loop_uni.py` | Self-loop removal functions for the unicycle cases. |

### System Dynamics Modules (`helpers/systems/`)

| Path | Role |
| --- | --- |
| `helpers/systems/synthetic.py` | Synthetic system dynamics. |
| `helpers/systems/mountain_car.py` | Mountain car dynamics and policy-dependent stepping logic. |
| `helpers/systems/unicycle.py` | Unicycle system dynamics. |
| `helpers/systems/policy.pth` | Policy weights used by the mountain car system helper. |

### Generated and Cached Artifacts

| Path | Role |
| --- | --- |
| `cache/` | Cached ground-truth evaluation results reused across runs. |
| `out/` | Generated figures and output artifacts organized by system (`synthetic`, `mountain_car`, `unicycle`). |

### Environment and Dependency Files

| Path | Role |
| --- | --- |
| `requirements.txt` | Python dependency specification for the repository. |
