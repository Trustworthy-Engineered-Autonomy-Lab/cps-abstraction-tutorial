'''
usage:
python run_cegar --case {synthetic, mountaincar, unicycle} --method {AABB, POLY}

python run_cegar.py --case synthetic --method AABB
python run_cegar.py --case synthetic --method POLY

python run_cegar.py --case mountaincar --method AABB
python run_cegar.py --case mountaincar --method POLY

python run_cegar.py --case unicycle --method AABB
python run_cegar.py --case unicycle --method POLY

'''
import argparse
import subprocess
import sys
import os

COMPARE_SCRIPT = "../src/cegar/compare_to_ground_truth.py"
UNICYCLE_SCRIPT = "../src/cegar/abstract_unicycle/run_unicycle_cegar.py"
UNI_GT_CACHE = "../src/cegar/abstract_unicycle/cache/unicycle_cfg_e5336e8c1848.pkl"

# synthetic / mountaincar defaults
NX = 100
NY = 100
BUDGET = 10
MAX_STEPS = 100
GT_RESOLUTION = 100
GT_MAX_STEPS = 100

# unicycle defaults
UNI_NX = 20
UNI_NY = 20
UNI_NZ = 20
UNI_SPLIT_BUDGET = 10000
UNI_MAX_ITERS = 2500
UNI_HORIZON = 100


def run_general(system, method):
    cmd = [
        sys.executable,
        COMPARE_SCRIPT,
        "--system", system,
        "--method", method,
        "--nx", str(NX),
        "--ny", str(NY),
        "--budget", str(BUDGET),
        "--max_steps", str(MAX_STEPS),
        "--gt_grid_resolution", str(GT_RESOLUTION),
        "--gt_max_steps", str(GT_MAX_STEPS),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "../src"

    print("\nRunning:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True, env=env)


def run_unicycle(method):
    if method == "AABB":
        init_method = "aabb"
        refine_method = "aabb"
    elif method == "POLY":
        init_method = "poly"
        refine_method = "aabb"
    else:
        raise ValueError("Invalid method")

    cmd = [
        sys.executable,
        UNICYCLE_SCRIPT,
        "--init-method", init_method,
        "--refine-method", refine_method,
        "--nx", str(UNI_NX),
        "--ny", str(UNI_NY),
        "--nz", str(UNI_NZ),
        "--split-budget", str(UNI_SPLIT_BUDGET),
        "--max-iters", str(UNI_MAX_ITERS),
        "--horizon", str(UNI_HORIZON),
        "--gt-cache", UNI_GT_CACHE,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "../src"

    print("\nRunning:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        choices=["synthetic", "mountaincar", "unicycle"],
        required=True,
    )

    parser.add_argument(
        "--method",
        choices=["AABB", "POLY"],
        required=True,
    )

    args = parser.parse_args()

    if args.case == "synthetic":
        run_general("helpers.systems.synthetic", args.method)
    elif args.case == "mountaincar":
        run_general("helpers.systems.mountain_car", args.method)
    elif args.case == "unicycle":
        run_unicycle(args.method)


if __name__ == "__main__":
    main()
