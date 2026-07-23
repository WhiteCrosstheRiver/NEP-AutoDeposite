#!/usr/bin/env python3
"""分析 param_sweep 结果：留存率 + 岛心 Ge 集中度，输出排名。"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(ROOT, "configs", "param_sweep_v007_theta5_antivel", "manifest.tsv")
RUN_ROOT = os.path.join(ROOT, "runs_param_sweep_v007_theta5_antivel")
OUT_TSV = os.path.join(RUN_ROOT, "analysis_ranking.tsv")
CENTER_R = 20.0  # Å，岛心半径
INJECT_EXPECT = 499


def parse_box_center(fixed_json_path):
    with open(fixed_json_path, encoding="utf-8") as f:
        meta = json.load(f)
    px, py = meta["points"][0]
    return float(px), float(py)


def count_ge_and_center(dump_path, cx, cy, box_x, box_y):
    with open(dump_path, encoding="utf-8") as f:
        n = int(f.readline().strip())
        ge_xy = []
        for _ in range(n):
            parts = f.readline().split()
            if parts[0] != "Ge":
                continue
            x, y = float(parts[1]), float(parts[2])
            dx = min(abs(x - cx), box_x - abs(x - cx))
            dy = min(abs(y - cy), box_y - abs(y - cy))
            r = (dx * dx + dy * dy) ** 0.5
            ge_xy.append((x, y, r))
    n_ge = len(ge_xy)
    if n_ge == 0:
        return 0, 0.0, 0.0, 0.0
    radii = [t[2] for t in ge_xy]
    mean_r = sum(radii) / n_ge
    frac_center = sum(1 for r in radii if r <= CENTER_R) / n_ge
    return n_ge, mean_r, frac_center, max(radii)


def bfs_removed_from_log(log_path):
    if not os.path.isfile(log_path):
        return None
    text = open(log_path, encoding="utf-8", errors="ignore").read()
    return sum(int(x) for x in re.findall(r"Removed (\d+) disconnected", text))


def analyze_one(run_subdir, ts, v, sigma, log_path=None):
    run_dir = os.path.join(ROOT, run_subdir)
    dump = os.path.join(run_dir, "500", "dump.xyz")
    fixed_json = os.path.join(run_dir, "fixed_inject_xy.json")
    rounds = len([d for d in os.listdir(run_dir) if d.isdigit()]) if os.path.isdir(run_dir) else 0

    if not os.path.isfile(dump):
        return {
            "status": "incomplete",
            "rounds": rounds,
            "ts": ts,
            "v": v,
            "sigma": sigma,
            "run_subdir": run_subdir,
        }

    cx, cy = parse_box_center(fixed_json)
    box_x = box_y = 391.9464
    n_ge, mean_r, frac_center, max_r = count_ge_and_center(dump, cx, cy, box_x, box_y)
    retention = n_ge / INJECT_EXPECT
    bfs_rem = bfs_removed_from_log(log_path) if log_path else None

    # 主目标：留存；次目标：岛心浓度（frac_center 高、mean_r 低）
    score = retention * (0.5 + 0.5 * frac_center)

    return {
        "status": "done",
        "rounds": rounds,
        "ts": ts,
        "v": v,
        "sigma": sigma,
        "run_subdir": run_subdir,
        "n_ge": n_ge,
        "retention": retention,
        "frac_center_20A": frac_center,
        "mean_r_xy": mean_r,
        "max_r_xy": max_r,
        "bfs_removed": bfs_rem,
        "score": score,
    }


def main():
    if not os.path.isfile(MANIFEST):
        print("Run generate_param_sweep_configs.py first", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            _id, ts, v, sigma, run_subdir, cfg = parts[:6]
            log_name = os.path.basename(run_subdir) + ".log"
            log_path = os.path.join(RUN_ROOT, "logs", log_name)
            rows.append(analyze_one(run_subdir, float(ts), float(v), float(sigma), log_path))

    done = [r for r in rows if r.get("status") == "done"]
    done.sort(key=lambda r: (-r["retention"], -r["frac_center_20A"], r["mean_r_xy"]))

    os.makedirs(RUN_ROOT, exist_ok=True)
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write(
            "rank\tstatus\tts\tv\tsigma\tn_ge\tretention\tfrac_center20A\tmean_r_xy\tbfs_removed\tscore\trun_subdir\n"
        )
        for i, r in enumerate(done, 1):
            f.write(
                f"{i}\t{r['status']}\t{r['ts']}\t{r['v']}\t{r['sigma']}\t{r['n_ge']}\t"
                f"{r['retention']:.4f}\t{r['frac_center_20A']:.4f}\t{r['mean_r_xy']:.2f}\t"
                f"{r.get('bfs_removed', '')}\t{r['score']:.4f}\t{r['run_subdir']}\n"
            )
        for r in rows:
            if r.get("status") != "done":
                f.write(
                    f"-\t{r.get('status','?')}\t{r['ts']}\t{r['v']}\t{r['sigma']}\t"
                    f"-\t-\t-\t-\t-\t-\t{r['run_subdir']}\n"
                )

    print(f"Analyzed {len(done)}/{len(rows)} completed runs")
    print(f"Ranking -> {OUT_TSV}\n")
    print("Top 10 (retention × center):")
    for i, r in enumerate(done[:10], 1):
        print(
            f"  {i}. ts={r['ts']} v={r['v']} sig={r['sigma']} "
            f"Ge={r['n_ge']} ret={r['retention']*100:.1f}% "
            f"center@{CENTER_R}Å={r['frac_center_20A']*100:.1f}% mean_r={r['mean_r_xy']:.1f}Å"
        )


if __name__ == "__main__":
    main()
