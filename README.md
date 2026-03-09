# A Pragmatic Guide to Building Conservative Discrete Abstractions of Cyber-Physical Systems

This repository provides the full implementation accompanying “A Pragmatic Guide to Building Conservative Discrete Abstractions of Cyber-Physical Systems.” The extended version of the paper is [available on arXiv](https://arxiv.org/). The workflow is demonstrated on three case studies: (1) a synthetic GES system, (2) the classical Mountain Car environment, and (3) a unicycle dynamical system. To support reproducibility, the repository includes the complete experimental codebase, smoketest scripts for reproducing the paper’s results, and a Docker configuration for a portable, reproducible execution environment.

## Project Structure

```text
.
├── Dockerfile                  # Reproducible container environment
├── README.md                   # Setup and usage instructions
├── requirements.txt            # Python dependencies
├── full-paper.pdf              # Extended version of the paper
├── scripts/                    # Entry points for reproducing experiments
│   ├── run_base.py             # Base abstraction pipelines
│   ├── run_selfloop.py         # Self-loop-removal pipelines
│   └── run_cegar.py            # CEGAR pipelines
├── runs/                       # Default output directory
│   ├── base/                   # Outputs from baseline runs
│   ├── selfloop/               # Outputs from self-loop-removal runs
│   └── cegar/                  # Outputs from CEGAR runs
└── src/                        # Core source code
    ├── base-pipelines/         # End-to-end base abstraction pipelines
    ├── selfloop-pipelines/     # End-to-end self-loop-removal pipelines
    ├── cegar/                  # CEGAR implementations
    ├── systems/                # Case-study dynamics and parameters
    └── utils/                  # Shared utilities for plotting, caching, etc.
```

## Requirements and Installation

### Virtual environment

1. **Python 3.10+**: 
Install Python 3.10 or later from your preferred package manager or [python.org](https://www.python.org/).

2. **Configure Environment**:

   Create and activate a virtual environment from the repo root:

   **Windows (PowerShell):**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux (bash/zsh):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   (Optional) Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Python Packages**: Install the required Python packages (listed in `requirements.txt`) using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Docker
The project is configured to be deployed using Docker for ease of setup and reproducibility. Follow these steps to build and run the Docker image:

1. **Build the Docker Image**:
   Navigate to the `artifact-xyz/` directory and run the following command to build the Docker image:
   ```bash
   docker build -t artifact-xyz -f artifact-xyz/Dockerfile .
   ```

2. **Run the Docker Container**:
   Once the image is built, you can run the container with:
   ```bash
   docker run -it artifact-xyz
   ```

3. **Access the Container**:
   To access the container's bash shell for debugging or manual execution, use:
   ```bash
   docker run -it artifact-xyz /bin/bash
   ```

4. **Rebuild the Image After Changes**:
   If you make changes to the project files or Dockerfile, rebuild the image.

## Running the Pipelines

These commands run the pipeline scripts in `scripts/`:

- build a fixed-grid abstraction for the chosen system, construct the Kripke structure, execute model checking, and then report the results for each transition building subroutine (AABB-, PT-, and sample-based):
    ```bash
    python -u scripts/run_base.py --case {synthetic,mountain_car,unicycle}
    ```

- the same pipeline as before, but now with self-loop erasure (reach-set and sample-based) for each transition building subroutine:
    ```bash
    python -u scripts/run_selfloop.py --case {synthetic,mountain_car,unicycle}
    ```

- the same base pipeline, but now with CEGAR (reach-set and sample-based) for each transition building subroutine:
    ```bash
    python -u scripts/run_cegar.py --case synthetic
    python -u scripts/run_cegar.py --case mountain_car --method POLY --nx 20 --ny 20
    python -u scripts/run_cegar.py --case unicycle
    ```

    Note: `scripts/run_cegar.py` defaults to `--method AABB` for speed. Use `--method POLY` if you specifically want the convex-hull transition builder (it can be much slower on large grids).

    Each run prints stage-wise metrics and saves `metrics.json` under `runs/cegar/<case>/` by default.

<!-- ### Base-pipeline dispatcher (recommended)

Each script defines an `ARGS` configuration dictionary in `__main__` for:

- grid resolution and domain parameters
- enabled transition methods (`run_aabb`, `run_poly`, `run_sample`)
- sampling parameters (`sample_n`, `sample_seed`)
- Kripke/model-checking controls
- ground-truth evaluation controls
- plotting controls

Additional `ARGS` in the self-loop-removal pipelines:

- max steps for reachable-set propagation certificate (`[method]_self_loop_max_steps`)
- number of samples for sample-based certificate (`[method]_sample_exit_n`)
- max steps per sample (`[method]_sample_exit_max_steps`)

By default, the dispatchers run pipelines from the repo root so relative paths land under `runs/`. Some plotting helpers also write figures to `out/` when enabled.

## Successor (Transition) Methods

The transition builders are implemented in `src/utils/partitioning.py`. All methods return a transition map with the same interface:

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

`SAMPLE` estimates successors empirically by drawing batches of states over the domain, binning sampled source and stepped destination states into grid cells, and recording observed `(source, destination)` pairs. Sampling terminates when a Good-Turing style missing-mass upper bound falls below `beta` (with confidence parameter `delta`). -->

