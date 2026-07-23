import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utility import sample_mbe_velocity


def sample_species(species: str, temperature_k: float, n_samples: int, direction_axis: str, seed: int):
    rng = np.random.default_rng(seed)
    velocities = np.array(
        [sample_mbe_velocity(species, temperature_k, direction_axis=direction_axis, rng=rng) for _ in range(n_samples)]
    )
    speed = np.linalg.norm(velocities, axis=1)
    return velocities, speed


def plot_distributions(species: str, velocities: np.ndarray, speed: np.ndarray, out_dir: Path):
    labels = ["vx", "vy", "vz"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    for idx, comp in enumerate(labels):
        axes[idx].hist(velocities[:, idx], bins=40, alpha=0.8, edgecolor="black")
        axes[idx].set_title(f"{species} {comp} (A/fs)")
        axes[idx].set_xlabel(comp)
        axes[idx].set_ylabel("count")
        axes[idx].grid(alpha=0.25)

    axes[3].hist(speed, bins=40, alpha=0.8, edgecolor="black", color="tab:orange")
    axes[3].set_title(f"{species} |v| (A/fs)")
    axes[3].set_xlabel("|v|")
    axes[3].set_ylabel("count")
    axes[3].grid(alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / f"velocity_distribution_{species}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def print_stats(species: str, velocities: np.ndarray, speed: np.ndarray):
    mean_v = velocities.mean(axis=0)
    std_v = velocities.std(axis=0)
    print(f"\n[{species}] 样本统计")
    print(f"  vx mean/std: {mean_v[0]: .6e} / {std_v[0]: .6e} A/fs")
    print(f"  vy mean/std: {mean_v[1]: .6e} / {std_v[1]: .6e} A/fs")
    print(f"  vz mean/std: {mean_v[2]: .6e} / {std_v[2]: .6e} A/fs")
    print(f"  |v| mean/std: {speed.mean(): .6e} / {speed.std(): .6e} A/fs")
    print(f"  vz < 0 比例: {(velocities[:, 2] < 0).mean():.3f}")


def main():
    parser = argparse.ArgumentParser(description="测试 sample_mbe_velocity 的速度分布并绘图")
    parser.add_argument("--temperature", type=float, default=1200.0, help="温度 (K)")
    parser.add_argument("--n", type=int, default=1000, help="每种元素采样数量")
    parser.add_argument("--direction", type=str, default="z", choices=["x", "y", "z"], help="束流主方向")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out-dir", type=str, default=".", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, species in enumerate(["Si", "Ge"]):
        velocities, speed = sample_species(
            species=species,
            temperature_k=args.temperature,
            n_samples=args.n,
            direction_axis=args.direction,
            seed=args.seed + idx,
        )
        print_stats(species, velocities, speed)
        out_path = plot_distributions(species, velocities, speed, out_dir=out_dir)
        print(f"  图像已保存: {out_path}")


if __name__ == "__main__":
    main()
