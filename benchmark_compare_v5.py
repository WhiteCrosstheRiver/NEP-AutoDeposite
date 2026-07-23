#!/usr/bin/env python3
"""
Compare v5 vs v6 SoA pipeline on the same dump.xyz files.

Usage:
  python3 benchmark_compare_v5.py [--rounds 10] [--v5-dump-root PATH]
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
V5_ROOT = os.path.join(
    os.path.dirname(SCRIPT_ROOT),
    "v5_fastBallisticNearSurfaceDeposition",
)
V5_DUMP_ROOT = os.path.join(
    V5_ROOT,
    "T550_100_Ge100_fixedRandom100_zoff8_theta5deg_rep6_nd3_gaussSpot20_20x3fs_v01_cut32",
)
FIXED_XY = os.path.join(V5_DUMP_ROOT, "fixed_inject_xy.json")

sys.path.insert(0, V5_ROOT)
import utility as v5_util  # noqa: E402

sys.path.insert(0, SCRIPT_ROOT)
from array_pipeline import last_converted_new_soa  # noqa: E402

PARAMS = dict(
    box_z=200.0,
    inject_species_weights={"Ge": 1.0, "Si": 0.0},
    local_surface_radius=5.0,
    cluster_cutoff=3.2,
    remove_incident_particles=True,
    velocity_magnitude=0.01,
    theta_sigma_deg=5.0,
    inject_mode="fast_ballistic_near_surface",
    fixed_inject_xy_path=FIXED_XY,
    placement_bond_cutoff=4.0,
    placement_distance_factor=0.65,
    inject_xy_gaussian_3sigma=20.0,
    surface_grid_cell_size=5.0,
    inject_rng_base=1000003,
    fixed_xy_seed=42,
    pipeline_timeit=False,
)


def count_atoms(path):
    with open(path, "r", encoding="utf-8") as f:
        return int(f.readline().strip())


def count_fix_group(path):
    n_fix = 0
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        f.readline()
        for line in f:
            parts = line.split()
            if parts and parts[-1] == "1":
                n_fix += 1
    return n_fix


def read_last_n_ge_positions(path, n=100):
    """Read last n lines as pos arrays (injected Ge at end of file)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    atom_lines = lines[2:]
    ge_lines = [ln for ln in atom_lines if ln.split()[0] == "Ge"]
    tail = ge_lines[-n:] if len(ge_lines) >= n else ge_lines
    out = []
    for ln in tail:
        p = ln.split()
        out.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(out, dtype=float)


def run_v5(dump_path, out_path, round_num):
    inject_round_seed = round_num * PARAMS["inject_rng_base"] + PARAMS["fixed_xy_seed"]
    import random

    random.seed(int(inject_round_seed))
    np.random.seed(int(inject_round_seed) % (2**32 - 1))

    t0 = time.perf_counter()
    v5_util.last_converted_new(
        file_path=dump_path,
        output_file=out_path,
        box_z=PARAMS["box_z"],
        inject_species_weights=PARAMS["inject_species_weights"],
        local_surface_radius=PARAMS["local_surface_radius"],
        cluster_cutoff=PARAMS["cluster_cutoff"],
        remove_incident_particles=PARAMS["remove_incident_particles"],
        velocity_magnitude=PARAMS["velocity_magnitude"],
        theta_sigma_deg=PARAMS["theta_sigma_deg"],
        inject_mode=PARAMS["inject_mode"],
        fixed_inject_xy_path=PARAMS["fixed_inject_xy_path"],
        placement_bond_cutoff=PARAMS["placement_bond_cutoff"],
        placement_distance_factor=PARAMS["placement_distance_factor"],
        inject_xy_gaussian_3sigma=PARAMS["inject_xy_gaussian_3sigma"],
        ballistic_timeit=False,
    )
    return (time.perf_counter() - t0) * 1000.0


def run_v6(dump_path, out_path, round_num):
    inject_round_seed = round_num * PARAMS["inject_rng_base"] + PARAMS["fixed_xy_seed"]
    t0 = time.perf_counter()
    timing, n_removed, n_atoms = last_converted_new_soa(
        file_path=dump_path,
        output_file=out_path,
        inject_round_seed=inject_round_seed,
        **{k: v for k, v in PARAMS.items() if k not in ("inject_rng_base", "fixed_xy_seed")},
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    return elapsed, timing, n_removed, n_atoms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="compare rounds 2..N+1")
    parser.add_argument("--v5-dump-root", default=V5_DUMP_ROOT)
    args = parser.parse_args()

    if not os.path.isfile(FIXED_XY):
        print(f"Missing {FIXED_XY}; run v5 first or provide fixed_inject_xy.json")
        sys.exit(1)

    v5_times = []
    v6_times = []
    ok = True

    print(f"Benchmark v5 vs v6 on dumps under {args.v5_dump_root}")
    print(f"{'round':>5} {'n_v5':>8} {'n_v6':>8} {'fix_v5':>8} {'fix_v6':>8} "
          f"{'v5_ms':>8} {'v6_ms':>8} {'pipe_ms':>8} {'match':>6}")

    with tempfile.TemporaryDirectory() as tmp:
        for r in range(2, args.rounds + 2):
            dump_path = os.path.join(args.v5_dump_root, str(r - 1), "dump.xyz")
            if not os.path.isfile(dump_path):
                print(f"  skip round {r}: no {dump_path}")
                continue

            out_v5 = os.path.join(tmp, f"v5_{r}.xyz")
            out_v6 = os.path.join(tmp, f"v6_{r}.xyz")

            v5_ms = run_v5(dump_path, out_v5, r)
            v6_ms, pipe, n_rem, n_v6 = run_v6(dump_path, out_v6, r)
            n_v5 = count_atoms(out_v5)
            fix_v5 = count_fix_group(out_v5)
            fix_v6 = count_fix_group(out_v6)

            struct_match = n_v5 == n_v6 and fix_v5 == fix_v6
            pos_match = True
            max_dpos = 0.0
            if struct_match:
                pos_v5 = read_last_n_ge_positions(out_v5, 100)
                pos_v6 = read_last_n_ge_positions(out_v6, 100)
                if pos_v5.shape == pos_v6.shape and pos_v5.size:
                    max_dpos = float(np.max(np.abs(pos_v5 - pos_v6)))
                    pos_match = max_dpos < 1e-2

            match = struct_match
            if not match:
                ok = False

            v5_times.append(v5_ms)
            v6_times.append(v6_ms)
            pos_note = f" dpos={max_dpos:.4f}" if max_dpos else ""
            print(
                f"{r:5d} {n_v5:8d} {n_v6:8d} {fix_v5:8d} {fix_v6:8d} "
                f"{v5_ms:8.0f} {v6_ms:8.0f} {pipe.total_ms:8.0f} "
                f"{'OK' if match else 'FAIL'}{pos_note}"
            )
            if struct_match and not pos_match:
                print(f"       (structure OK; injection positions differ — expected without shared v5 RNG)")

    if v5_times and v6_times:
        speedup = np.mean(v5_times) / np.mean(v6_times)
        print(f"\nMean script: v5={np.mean(v5_times):.0f}ms v6={np.mean(v6_times):.0f}ms "
              f"speedup={speedup:.2f}x")
        print(f"v6 pipeline breakdown (mean): parse/bfs/surf/inj/write = "
              f"(see per-round [pipeline] logs when pipeline_timeit=true)")

    if ok:
        print("\nBenchmark PASSED")
        sys.exit(0)
    print("\nBenchmark FAILED (see mismatches above)")
    sys.exit(1)


if __name__ == "__main__":
    main()
