"""v7：衬底正中央单点固定 XY（不修改 v6 utility）。"""

import os

from utility import load_fixed_inject_xy, parse_box_xy_from_xyz, save_fixed_inject_xy


def ensure_center_fixed_inject_xy(model_xyz_path, json_path, seed=42):
    """
    若 JSON 不存在则在 box 正中央写入 1 个固定注入点；
    若已存在则直接加载。返回 json_path。
    """
    if os.path.isfile(json_path):
        points, meta = load_fixed_inject_xy(json_path)
        print(
            f"Loaded {len(points)} center inject XY from {json_path} "
            f"(seed={meta.get('seed')})"
        )
        for i, (px, py) in enumerate(points, start=1):
            print(f"  point {i}: ({px:.3f}, {py:.3f})")
        return json_path

    box_x, box_y = parse_box_xy_from_xyz(model_xyz_path)
    px = float(box_x) / 2.0
    py = float(box_y) / 2.0
    points = [(px, py)]
    save_fixed_inject_xy(json_path, box_x, box_y, seed, 1, points)
    print(
        f"Center inject XY ({px:.3f}, {py:.3f}), "
        f"box={box_x:.3f}x{box_y:.3f} -> {json_path}"
    )
    return json_path
