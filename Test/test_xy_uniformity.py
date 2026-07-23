"""
测试粒子表面入射算法在 XY 平面上的初始位置是否平铺均匀。

模拟真实生产流程：
  每轮调用 generate_particle_positions 固定撒 batch_size 个点（默认 4，与 injection_count 一致），
  轮与轮之间不共享“已放置”列表（与 demo.py 每轮重新注入一致），
  重复多轮直到累计 n_total 个点，再统计 XY 分布。
"""
import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SingleRunInputScript"))
from utility import _pbc_delta  # noqa: E402


def generate_particle_xy_batch(
    box_x,
    box_y,
    target_particles,
    min_dist,
    max_attempts,
    rng,
):
    """
    与 utility.generate_particle_positions 的 XY 拒绝采样完全一致；
    Z 由表面高度决定，不影响 (px, py) 的抽样，故测试平铺均匀性时省略 Z 以加速。
    """
    positions = []
    attempts = 0
    target_particles = max(0, int(target_particles))
    min_dist_sq = float(min_dist) ** 2

    while len(positions) < target_particles and attempts < max_attempts:
        attempts += 1
        px = float(rng.uniform(0.0, box_x))
        py = float(rng.uniform(0.0, box_y))

        conflict = False
        for ex, ey in positions:
            dx = _pbc_delta(px, ex, box_x)
            dy = _pbc_delta(py, ey, box_y)
            if dx * dx + dy * dy < min_dist_sq:
                conflict = True
                break
        if not conflict:
            positions.append((px, py))

    return positions


def load_substrate(xyz_path: Path):
    with open(xyz_path, "r") as f:
        lines = f.readlines()
    lattice_str = lines[1]
    box_match = re.search(r'Lattice="([^"]*)"', lattice_str)
    if not box_match:
        raise ValueError(f"无法解析 Lattice: {lattice_str.strip()}")
    box_params = list(map(float, box_match.group(1).split()))
    box_x, box_y = box_params[0], box_params[4]
    positions = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(positions, dtype=float), box_x, box_y


def sample_by_rounds(
    box_x,
    box_y,
    n_total,
    batch_size,
    min_dist,
    seed,
    max_attempts=2000,
    progress_every=500,
):
    """
    每轮撒 batch_size 个点，累计至 n_total（轮与轮独立，同生产注入）。
    返回 xy 数组及每轮实际放置数量统计。
    """
    rng = np.random.default_rng(seed)
    xy_list = []
    placed_per_round = []
    round_idx = 0
    t0 = time.time()

    while len(xy_list) < n_total:
        round_idx += 1
        pts = generate_particle_xy_batch(
            box_x=box_x,
            box_y=box_y,
            target_particles=batch_size,
            min_dist=min_dist,
            max_attempts=max_attempts,
            rng=rng,
        )
        n_placed = len(pts)
        placed_per_round.append(n_placed)
        for px, py in pts:
            xy_list.append((px, py))
            if len(xy_list) >= n_total:
                break

        if progress_every and round_idx % progress_every == 0:
            elapsed = time.time() - t0
            print(
                f"  已完成 {round_idx} 轮, 累计 {len(xy_list)} 点 "
                f"(最近一轮放置 {n_placed}/{batch_size}), 耗时 {elapsed:.1f}s"
            )

    xy = np.asarray(xy_list[:n_total], dtype=float)
    meta = {
        "n_rounds": round_idx,
        "placed_per_round": np.asarray(placed_per_round, dtype=int),
        "elapsed_s": time.time() - t0,
    }
    return xy, meta


def chi_square_uniformity(xy, box_x, box_y, n_bins=20):
    hist, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=n_bins, range=[[0, box_x], [0, box_y]]
    )
    expected = xy.shape[0] / (n_bins * n_bins)
    chi2 = np.sum((hist - expected) ** 2 / expected)
    dof = n_bins * n_bins - 1
    return chi2, dof, hist


def plot_results(xy, box_x, box_y, batch_size, min_dist, meta, out_dir: Path):
    chi2, dof, hist2d = chi_square_uniformity(xy, box_x, box_y)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.35, c="tab:blue", edgecolors="none")
    ax.set_xlim(0, box_x)
    ax.set_ylim(0, box_y)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(
        f"XY 散点 (累计 {len(xy)} 点)\n"
        f"每轮 {batch_size} 点 × {meta['n_rounds']} 轮, min_dist={min_dist} Å"
    )
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    im = ax.imshow(
        hist2d.T,
        origin="lower",
        extent=[0, box_x, 0, box_y],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(f"2D 直方图 (20×20 格)\nχ²={chi2:.1f}, dof={dof} (≈{dof} 为均匀)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col, comp, box_len in [(0, 0, box_x), (1, 1, box_y)]:
        ax = axes[1, col]
        nb = 30
        edges = np.linspace(0, box_len, nb + 1)
        counts, _ = np.histogram(xy[:, comp], bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        expected = len(xy) / nb
        label = "x" if comp == 0 else "y"
        ax.bar(centers, counts, width=edges[1] - edges[0], alpha=0.8, color="tab:blue")
        ax.axhline(expected, color="tab:red", ls="--", lw=1.5, label=f"均匀期望 {expected:.0f}")
        ax.set_xlabel(f"{label} (Å)")
        ax.set_ylabel("count")
        ax.set_title(f"{label.upper()} 边缘分布")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"入射粒子 XY 平铺均匀性  |  box={box_x:.2f}×{box_y:.2f} Å",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    out_main = out_dir / "xy_uniformity_rounds.png"
    fig.savefig(out_main, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 每轮实际放置数分布（检查是否经常放不满）
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    vals, cnts = np.unique(meta["placed_per_round"], return_counts=True)
    ax2.bar(vals, cnts, width=0.6, color="tab:orange", edgecolor="black")
    ax2.set_xlabel("每轮实际放置数")
    ax2.set_ylabel("轮数")
    ax2.set_title(f"每轮放置数统计 (目标 {batch_size}/轮)")
    ax2.set_xticks(vals)
    ax2.grid(axis="y", alpha=0.3)
    out_rounds = out_dir / "xy_uniformity_placed_per_round.png"
    fig2.savefig(out_rounds, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return out_main, out_rounds, chi2, dof


def print_stats(xy, box_x, box_y, batch_size, min_dist, meta, chi2, dof):
    mean_x, mean_y = xy.mean(axis=0)
    std_x, std_y = xy.std(axis=0)
    ux, uy = box_x / 2, box_y / 2
    us = box_x / np.sqrt(12)
    ppr = meta["placed_per_round"]

    print("\n========== 采样汇总 ==========")
    print(f"  总点数:     {len(xy)}")
    print(f"  总轮数:     {meta['n_rounds']}")
    print(f"  每轮目标:   {batch_size} 点")
    print(f"  min_dist:   {min_dist} Å")
    print(f"  总耗时:     {meta['elapsed_s']:.1f} s")
    print(f"  每轮放置:   mean={ppr.mean():.2f}, min={ppr.min()}, max={ppr.max()}")
    if ppr.min() < batch_size:
        short = int(np.sum(ppr < batch_size))
        print(f"  未满轮数:   {short} / {len(ppr)} ({100*short/len(ppr):.1f}%)")

    print("\n========== 均匀性统计 ==========")
    print(f"  x: mean={mean_x:.4f} (期望 {ux:.4f}), std={std_x:.4f} (期望 {us:.4f})")
    print(f"  y: mean={mean_y:.4f} (期望 {uy:.4f}), std={std_y:.4f} (期望 {us:.4f})")
    print(f"  卡方(20×20): χ²={chi2:.2f}, dof={dof}  (χ²≈dof 表示格点计数接近均匀)")


def main():
    parser = argparse.ArgumentParser(
        description="每轮固定撒若干点，多轮累计后测试 XY 平铺均匀性"
    )
    parser.add_argument("--xyz", type=str, default="../Substrate/100.xyz")
    parser.add_argument("--n", type=int, default=10000, help="累计总点数")
    parser.add_argument("--batch-size", type=int, default=4, help="每轮入射粒子数")
    parser.add_argument("--min-dist", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="test_output")
    parser.add_argument("--progress-every", type=int, default=500, help="每多少轮打印进度")
    args = parser.parse_args()

    test_dir = Path(__file__).resolve().parent
    xyz_path = (test_dir / args.xyz).resolve()
    out_dir = (test_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_positions, box_x, box_y = load_substrate(xyz_path)
    n_rounds_need = int(np.ceil(args.n / args.batch_size))
    print(f"衬底: {xyz_path.name}, 原子数={len(base_positions)} (仅用于读取 box)")
    print(f"box: {box_x:.4f} × {box_y:.4f} Å")
    print(
        f"计划: 每轮 {args.batch_size} 点, 约 {n_rounds_need} 轮 → 累计 {args.n} 点 "
        f"(min_dist={args.min_dist} Å)"
    )
    print("开始采样...")

    xy, meta = sample_by_rounds(
        box_x,
        box_y,
        n_total=args.n,
        batch_size=args.batch_size,
        min_dist=args.min_dist,
        seed=args.seed,
        progress_every=args.progress_every,
    )

    out_main, out_rounds, chi2, dof = plot_results(
        xy, box_x, box_y, args.batch_size, args.min_dist, meta, out_dir
    )
    print_stats(xy, box_x, box_y, args.batch_size, args.min_dist, meta, chi2, dof)
    print(f"\n图像: {out_main}")
    print(f"      {out_rounds}")


if __name__ == "__main__":
    main()
