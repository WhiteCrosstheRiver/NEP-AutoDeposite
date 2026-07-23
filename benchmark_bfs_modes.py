#!/usr/bin/env python3
"""
Compare full BFS vs candidate-shell light BFS on v5 dump.xyz files.

Usage:
  python3 benchmark_bfs_modes.py [--rounds 1,10,50,99,299] [--percentiles 85,90,95]
"""

import argparse
import os
import sys
import time

import numpy as np

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
V5_DUMP_ROOT = os.path.join(
    os.path.dirname(SCRIPT_ROOT),
    "v5_fastBallisticNearSurfaceDeposition",
    "T550_100_Ge100_fixedRandom100_zoff8_theta5deg_rep6_nd3_gaussSpot20_20x3fs_v01_cut32",
)

sys.path.insert(0, SCRIPT_ROOT)
from array_pipeline import parse_dump_soa  # noqa: E402
from utility import (  # noqa: E402
    _find_main_component_candidate_shell,
    _find_main_component_from_min_z,
    _removed_z_diagnostics,
    format_bfs_light_log,
    resolve_bfs_keep_mask,
)

CUTOFF = 3.2


def _time_full(pos, box_x, box_y, warmup=False):
    if warmup:
        _find_main_component_from_min_z(pos, box_x, box_y, CUTOFF)
    t0 = time.perf_counter()
    mask = _find_main_component_from_min_z(pos, box_x, box_y, CUTOFF)
    return mask, (time.perf_counter() - t0) * 1000.0


def _time_light(pos, box_x, box_y, percentile, warmup=False):
    if warmup:
        _find_main_component_candidate_shell(pos, box_x, box_y, CUTOFF, percentile=percentile)
    t0 = time.perf_counter()
    mask, stats = _find_main_component_candidate_shell(
        pos, box_x, box_y, CUTOFF, percentile=percentile
    )
    return mask, stats, (time.perf_counter() - t0) * 1000.0


def _compare_masks(full_mask, light_mask):
    false_delete = int(np.sum(full_mask & ~light_mask))
    false_keep = int(np.sum(~full_mask & light_mask))
    match = bool(np.array_equal(full_mask, light_mask))
    return false_delete, false_keep, match


def benchmark_round(round_num, dump_root, percentiles, interval):
    dump_path = os.path.join(dump_root, str(round_num), "dump.xyz")
    if not os.path.isfile(dump_path):
        print(f"  skip round {round_num}: {dump_path} not found")
        return None

    frame, box_x, box_y, _ = parse_dump_soa(dump_path, fast_io=True)
    pos = frame.pos
    n = pos.shape[0]

    _time_full(pos, box_x, box_y, warmup=True)
    full_mask, full_ms = _time_full(pos, box_x, box_y)
    full_removed = int((~full_mask).sum())
    rz = _removed_z_diagnostics(pos[:, 2], full_mask)
    z_p90 = float(np.percentile(pos[:, 2], 90))

    print(f"\n=== round {round_num}  N={n}  full_removed={full_removed}  full_time={full_ms:.1f}ms ===")
    print(
        f"  full removed z: min={rz[0]:.2f} p10={rz[1]:.2f} p50={rz[2]:.2f} max={rz[3]:.2f}  "
        f"(z_p90={z_p90:.2f})"
    )

    row = {
        "round": round_num,
        "n": n,
        "full_ms": full_ms,
        "full_removed": full_removed,
        "removed_z_min": rz[0],
        "z_p90": z_p90,
        "percentiles": {},
    }

    for p in percentiles:
        _time_light(pos, box_x, box_y, p, warmup=True)
        light_mask, stats, light_ms = _time_light(pos, box_x, box_y, p)
        fd, fk, match = _compare_masks(full_mask, light_mask)
        light_removed = int((~light_mask).sum())

        avg_light = light_ms
        if interval > 1:
            # 1 full + (interval-1) light per block
            avg_light = (full_ms + (interval - 1) * light_ms) / interval

        print(
            f"  p{int(p):02d}: candidate={stats.n_candidate} attached={stats.n_attached_seed} "
            f"visited={stats.n_visited_candidate} light_removed={light_removed} "
            f"false_del={fd} false_keep={fk} match={match} "
            f"light={light_ms:.1f}ms avg(interval={interval})={avg_light:.1f}ms"
        )
        row["percentiles"][p] = {
            "light_ms": light_ms,
            "avg_ms": avg_light,
            "false_delete": fd,
            "false_keep": fk,
            "match": match,
            "light_removed": light_removed,
            "n_candidate": stats.n_candidate,
        }

    # shadow via resolve_bfs_keep_mask (uses p90 by default in caller)
    for p in percentiles:
        resolve_bfs_keep_mask(
            pos, box_x, box_y, CUTOFF,
            bfs_mode="shadow", bfs_light_percentile=p, round_num=round_num,
        )
    _, shadow_stats = resolve_bfs_keep_mask(
        pos, box_x, box_y, CUTOFF,
        bfs_mode="shadow", bfs_light_percentile=90.0, round_num=round_num,
    )
    print(f"  shadow log (p90): {format_bfs_light_log(shadow_stats, round_num, n)}")

    # light mode interval simulation for round_num
    _, light_stats = resolve_bfs_keep_mask(
        pos, box_x, box_y, CUTOFF,
        bfs_mode="light", bfs_full_interval=interval,
        bfs_light_percentile=90.0, round_num=round_num,
    )
    mode_label = light_stats.mode
    print(f"  resolve light round={round_num} interval={interval} -> mode={mode_label}")

    return row


def main():
    parser = argparse.ArgumentParser(description="Benchmark full vs candidate-shell BFS")
    parser.add_argument("--rounds", default="1,10,50,99,299")
    parser.add_argument("--percentiles", default="85,90,95")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--dump-root", default=V5_DUMP_ROOT)
    args = parser.parse_args()

    rounds = [int(x) for x in args.rounds.split(",") if x.strip()]
    percentiles = [float(x) for x in args.percentiles.split(",") if x.strip()]

    print(f"dump_root: {args.dump_root}")
    print(f"cluster_cutoff={CUTOFF}  interval={args.interval}")

    all_rows = []
    for r in rounds:
        row = benchmark_round(r, args.dump_root, percentiles, args.interval)
        if row is not None:
            all_rows.append(row)

    if not all_rows:
        print("No rounds benchmarked.")
        return 1

    print("\n=== SUMMARY ===")
    for p in percentiles:
        fd_sum = sum(row["percentiles"][p]["false_delete"] for row in all_rows)
        fk_sum = sum(row["percentiles"][p]["false_keep"] for row in all_rows)
        match_all = all(row["percentiles"][p]["match"] for row in all_rows)
        avg_full = np.mean([row["full_ms"] for row in all_rows])
        avg_light = np.mean([row["percentiles"][p]["light_ms"] for row in all_rows])
        avg_interval = np.mean([row["percentiles"][p]["avg_ms"] for row in all_rows])
        print(
            f"p{p:.0f}: match_all={match_all} total_false_delete={fd_sum} "
            f"total_false_keep={fk_sum} "
            f"mean_full={avg_full:.1f}ms mean_light={avg_light:.1f}ms "
            f"mean_avg(interval={args.interval})={avg_interval:.1f}ms"
        )

    removed_z_mins = [row["removed_z_min"] for row in all_rows if row["full_removed"] > 0]
    if removed_z_mins:
        print(
            f"removed_z_min across rounds with deletions: "
            f"min={min(removed_z_mins):.2f} max={max(removed_z_mins):.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
