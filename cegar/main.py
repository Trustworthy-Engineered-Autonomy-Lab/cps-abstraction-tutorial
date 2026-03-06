from __future__ import annotations

from abstraction import Rect, RectPartition
from cegar_loop import run_cegar
from krish_abstraction import KrishAbstraction
from helpers.systems.synthetic import SyntheticSystem

def main():
    domain = Rect(-10.0, 10.0, -10.0, 10.0)
    nx, ny = 40, 40
    part = RectPartition.uniform_grid(domain, nx, ny)
    system = SyntheticSystem()

    # Transition builder: "AABB", "POLY", or "SAMPLE"
    absys = KrishAbstraction(part=part, system=system, method="POLY")

    phi = "A (safe U goal)"

    init_uids = set(part.leaves.keys())

    res = run_cegar(
        absys,
        init_uids,
        phi,
        max_iters=25,
        max_steps_proxy=200,
        min_cell_width=0.0,
        min_cell_height=0.0,
        max_refine_depth=None,
        verbose=True,
    )
    print(res)

if __name__ == "__main__":
    main()
