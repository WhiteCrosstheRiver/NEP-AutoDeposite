#!/usr/bin/env python3
"""Benchmark Python vs C++ I/O paths on v5 dump files."""

import argparse
import os
import sys
import time

import numpy as np

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_ROOT)

from array_pipeline import (  # noqa: E402
    _HAS_FAST_XYZ,
    last_converted_new_soa,
    parse_dump_soa,
    write_model_xyz,
    _build_model_header,
)

V5_DUMP_ROOT = os.path.join(
    os.path.dirname(SCRIPT_ROOT),
    "v5_fastBallisticNearSurfaceDeposition",
    "T550_100_Ge100_fixedRandom100_zoff8_theta5deg_rep6_nd3_gaussSpot20_20x3fs_v01_cut32",
)
FIXED_XY = os.path.join(V5_DUMP_ROOT, "fixed_inject_xy.json")


def bench_round(dump_path, round_num, fast_io, decimals):
    out = f"/tmp/bench_io_r{round_num}_{'cpp' if fast_io else 'py'}.xyz"
    seed = round_num * 1000003 + 42
    t0 = time.perf_counter()
    timing, _, n_atoms = last_converted_new_soa(
        dump_path,
        out,
        fixed_inject_xy_path=FIXED_XY,
        inject_mode="fast_ballistic_near_surface",
        inject_round_seed=seed,
        pipeline_timeit=False,
        fast_io=fast_io,
        xyz_float_decimals=decimals,
    )
    total = (time.perf_counter() - t0) * 1000.0
    return timing, total, n_atoms, out


def compare_files(path_a, path_b):
    with open(path_a) as fa, open(path_b) as fb:
        la = fa.readlines()
        lb = fb.readlines()
    if len(la) != len(lb):
        return False, f"line count {len(la)} vs {len(lb)}"
    if la[0] != lb[0] or la[1] != lb[1]:
        return la[0] == lb[0] and la[1] == lb[1], "header mismatch"
    # compare numeric columns (allow fmt diff)
    for i, (a, b) in enumerate(zip(la[2:], lb[2:]), start=2):
        pa, pb = a.split(), b.split()
        if pa[0] != pb[0] or pa[-1] != pb[-1]:
            return False, f"line {i} species/group"
        fa = list(map(float, pa[1:7]))
        fb = list(map(float, pb[1:7]))
        if max(abs(x - y) for x, y in zip(fa, fb)) > 0.0005:
            return False, f"line {i} numeric"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    if not _HAS_FAST_XYZ:
        print("fast_xyz not built; run ./build_fast_xyz.sh first")
        sys.exit(1)

    print(f"fast_xyz available. rounds=2..{args.rounds + 1}")
    print(f"{'round':>5} {'py_ms':>8} {'cpp_ms':>8} {'parse_py':>8} {'parse_cpp':>9} "
          f"{'write_py':>8} {'write_cpp':>9} {'match':>6}")

    py_parse, py_write, py_total = [], [], []
    cpp_parse, cpp_write, cpp_total = [], [], []

    for r in range(2, args.rounds + 2):
        dump_path = os.path.join(V5_DUMP_ROOT, str(r - 1), "dump.xyz")
        if not os.path.isfile(dump_path):
            continue
        t_py, tot_py, _, out_py = bench_round(dump_path, r, fast_io=False, decimals=3)
        t_cpp, tot_cpp, _, out_cpp = bench_round(dump_path, r, fast_io=True, decimals=3)
        ok, msg = compare_files(out_py, out_cpp)
        py_parse.append(t_py.parse_ms)
        py_write.append(t_py.write_ms)
        py_total.append(tot_py)
        cpp_parse.append(t_cpp.parse_ms)
        cpp_write.append(t_cpp.write_ms)
        cpp_total.append(tot_cpp)
        print(
            f"{r:5d} {tot_py:8.0f} {tot_cpp:8.0f} {t_py.parse_ms:8.0f} {t_cpp.parse_ms:9.0f} "
            f"{t_py.write_ms:8.0f} {t_cpp.write_ms:9.0f} {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            print(f"       compare: {msg}")

    if py_total:
        print(
            f"\nMean pipeline: python={np.mean(py_total):.0f}ms cpp={np.mean(cpp_total):.0f}ms "
            f"speedup={np.mean(py_total)/np.mean(cpp_total):.2f}x"
        )
        print(
            f"Mean parse: py={np.mean(py_parse):.0f} cpp={np.mean(cpp_parse):.0f}ms | "
            f"write: py={np.mean(py_write):.0f} cpp={np.mean(cpp_write):.0f}ms"
        )


if __name__ == "__main__":
    main()
