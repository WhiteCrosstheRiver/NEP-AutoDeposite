#!/usr/bin/env python3
"""
将 GPUMD extended xyz 在 XY 平面做 nx×ny 复制（默认 2×2），Z 与 box_z 不变。
CLI 封装；核心逻辑在 SingleRunInputScript/utility.py。
"""
import argparse
import os
import sys

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SCRIPT_ROOT, "SingleRunInputScript"))
from utility import replicate_xy  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Substrate XYZ XY replicate")
    parser.add_argument("--src", default="Substrate", help="源目录")
    parser.add_argument("--dst", default="Substrate_replicate222", help="输出目录")
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if not os.path.isdir(src):
        print(f"源目录不存在: {src}", file=sys.stderr)
        sys.exit(1)

    xyz_files = sorted(f for f in os.listdir(src) if f.endswith(".xyz"))
    if not xyz_files:
        print(f"未找到 xyz: {src}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(dst, exist_ok=True)
    print(f"XY replicate {args.nx}x{args.ny}: {src} -> {dst}\n")
    for name in xyz_files:
        info = replicate_xy(
            os.path.join(src, name),
            os.path.join(dst, name),
            nx=args.nx,
            ny=args.ny,
        )
        print(
            f"{name}: {info['n_in']} -> {info['n_out']} atoms, "
            f"box {info['box_in'][0]:.4f}x{info['box_in'][1]:.4f} -> "
            f"{info['box_out'][0]:.4f}x{info['box_out'][1]:.4f} (Z={info['box_out'][2]:.1f})"
        )


if __name__ == "__main__":
    main()
