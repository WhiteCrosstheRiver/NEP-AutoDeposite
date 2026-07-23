"""
测试新的"腔体粒子模拟"算法：
1. XY 网格取样 + Z 均匀分布 — 空间均匀性 / PBC 随机性
2. 速度方向 — θ 正态, φ 均匀
"""
import sys, os, math, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SingleRunInputScript'))
from utility import get_local_surface_z_from_atoms

BOX_X, BOX_Y = 65.3244, 65.3244
SPACING = 14.0
Z_SPREAD = 50.0
SURFACE_HEIGHT_OFFSET = 8.0
MIN_PAIR_DIST = 13.0
N_PARTICLES = 100
VELOCITY_MAG = 0.01  # Å/fs
THETA_SIGMA_DEG = 5.0  # θ 标准差（度）


def generate_positions_grid_zsample(box_x, box_y, spacing, z_spread, n_target,
                                    surface_z=0.0, height_offset=SURFACE_HEIGHT_OFFSET,
                                    rng=None):
    """
    XY：细网格 + 每轮随机平移网格起点 offset ∈ [0, spacing)。
    这样每轮的 XY 分布均匀（覆盖 box）但位置不同。
    Z 在 [base_z, base_z + z_spread] 均匀随机。
    """
    if rng is None:
        rng = np.random.default_rng()
    nx = int(np.floor(box_x / spacing))
    ny = int(np.floor(box_y / spacing))
    base_z = surface_z + height_offset

    # 网格格心 + 随机平移
    offset_x = rng.uniform(0, float(spacing))
    offset_y = rng.uniform(0, float(spacing))
    xs = (np.arange(nx, dtype=float) + 0.5) * spacing + offset_x
    ys = (np.arange(ny, dtype=float) + 0.5) * spacing + offset_y
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid_positions = np.column_stack([xx.ravel(), yy.ravel()])
    n_grid = len(grid_positions)

    # 随机排列
    perm = rng.permutation(n_grid)
    grid_positions = grid_positions[perm]

    if n_target <= n_grid:
        xy = grid_positions[:n_target]
    else:
        extra = n_target - n_grid
        xy_extra = np.column_stack([
            rng.uniform(0, box_x, size=extra),
            rng.uniform(0, box_y, size=extra),
        ])
        xy = np.vstack([grid_positions, xy_extra])

    zs = rng.uniform(base_z, base_z + z_spread, size=n_target)
    positions = np.column_stack([xy[:, 0], xy[:, 1], zs])
    positions[:, 0] = np.mod(positions[:, 0], box_x)
    positions[:, 1] = np.mod(positions[:, 1], box_y)
    return positions


def test_spatial_uniformity():
    """测试 XY 空间均匀性 + Z 均匀分布"""
    print("=" * 60)
    print("1. 空间均匀性测试")
    print("=" * 60)

    n_trials = 10

    for label, n_particles, spacing in [("满网格 (64 from 8x8)", 64, 8.0),
                                         ("半网格 (20 from 4x4)", 20, BOX_X/4),
                                         ("2x网格 (100 from 4x4+随机)", 100, 14.0)]:
        positions_list = []
        rng = np.random.default_rng()
        for trial in range(n_trials):
            pos = generate_positions_grid_zsample(
                BOX_X, BOX_Y, spacing, Z_SPREAD, n_particles,
                surface_z=0.0, height_offset=SURFACE_HEIGHT_OFFSET,
                rng=rng,
            )
            positions_list.append(pos)

        all_positions = np.vstack(positions_list)

        # 多轮叠加 XY 均匀性
        nbins = 8
        hist, _, _ = np.histogram2d(
            all_positions[:, 0], all_positions[:, 1], bins=nbins,
            range=[[0, BOX_X], [0, BOX_Y]]
        )
        expected = len(all_positions) / (nbins * nbins)
        chi2_accum = np.sum((hist - expected)**2 / expected)

        # 单轮平均 chi2
        trial_chi2 = []
        for t in range(n_trials):
            h, _, _ = np.histogram2d(
                positions_list[t][:, 0], positions_list[t][:, 1],
                bins=4, range=[[0, BOX_X], [0, BOX_Y]]
            )
            e = n_particles / 16
            trial_chi2.append(np.sum((h - e)**2 / e))

        # Z 均匀性
        z_min = SURFACE_HEIGHT_OFFSET
        z_max = SURFACE_HEIGHT_OFFSET + Z_SPREAD
        z_hist, _ = np.histogram(all_positions[:, 2], bins=10, range=[z_min, z_max])
        z_expected = len(all_positions) / 10
        z_chi2 = np.sum((z_hist - z_expected)**2 / z_expected)

        print(f"\n  [{label}]")
        print(f"    单轮平均 chi2: {np.mean(trial_chi2):.1f}")
        print(f"    {n_trials}轮叠加 chi2: {chi2_accum:.1f} (越低越均匀)")
        print(f"    Z 分布 chi2: {z_chi2:.1f}")


def test_pbc_randomness():
    """测试每一轮 XY 是否从网格中随机抽取（不是固定取前 N 个）"""
    print("\n" + "=" * 60)
    print("2. PBC 随机性测试（每轮抽取的网格索引不同）")
    print("=" * 60)

    n_trials = 5
    n_particles = 20  # 从 4x4=16 网格中抽 20（16网格 + 4 random）
    spacing = BOX_X / 4

    indices_sets = []
    for trial in range(n_trials):
        pos = generate_positions_grid_zsample(
            BOX_X, BOX_Y, spacing, Z_SPREAD, n_particles,
            surface_z=0.0, height_offset=SURFACE_HEIGHT_OFFSET
        )
        grid_xs = (np.arange(4) + 0.5) * spacing
        # 记录哪个粒子在格心上（近似容差）
        on_grid_mask = np.array([
            min(abs(px - gx) for gx in grid_xs) < 0.01
            for px in pos[:, 0]
        ])
        n_on_grid = np.sum(on_grid_mask)
        n_random = n_particles - n_on_grid
        if trial < 3:
            print(f"  Trial {trial+1}: grid={n_on_grid}, random_XY={n_random}")
        indices_sets.append(n_on_grid)

    print(f"  每轮从网格抽取数量: {indices_sets}")
    all_same = len(set(indices_sets)) == 1
    print(f"  {'警告：每轮相同！' if all_same else 'OK：每轮随机抽取不同'}")


def test_min_distance_performance():
    """测试拒绝采样的时间开销"""
    print("\n" + "=" * 60)
    print("3. 性能测试（含 min_pair_dist 拒绝采样）")
    print("=" * 60)

    def generate_with_rejection(box_x, box_y, spacing, z_spread, n_target,
                                 min_dist=13.0, surface_z=0.0, height_offset=8.0,
                                 max_attempts=10000):
        """
        先网格取 XY + Z 均匀，然后做三维拒绝采样确保间距 >= min_dist。
        """
        nx = int(np.floor(box_x / spacing))
        ny = int(np.floor(box_y / spacing))
        base_z = surface_z + height_offset

        xs = (np.arange(nx, dtype=float) + 0.5) * spacing
        ys = (np.arange(ny, dtype=float) + 0.5) * spacing
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        grid_pool = np.column_stack([xx.ravel(), yy.ravel()])
        pool_size = len(grid_pool)
        rng = np.random.default_rng()

        # 先随机排序网格
        perm = rng.permutation(pool_size)
        grid_pool = grid_pool[perm]

        positions = []
        attempts = 0
        g_idx = 0  # grid index
        while len(positions) < n_target and attempts < max_attempts:
            attempts += 1
            # XY: 有网格就用网格，无网格纯随机
            if g_idx < pool_size:
                px, py = grid_pool[g_idx]
                g_idx += 1
            else:
                px = rng.uniform(0, box_x)
                py = rng.uniform(0, box_y)
            pz = rng.uniform(base_z, base_z + z_spread)

            # PBC wrap
            px = np.mod(px, box_x)
            py = np.mod(py, box_y)

            # 间距检查
            if min_dist > 0:
                conflict = False
                for ex, ey, ez in positions:
                    dx = abs(px - ex)
                    dx = min(dx, box_x - dx)
                    dy = abs(py - ey)
                    dy = min(dy, box_y - dy)
                    dz = pz - ez
                    if dx*dx + dy*dy + dz*dz < min_dist*min_dist:
                        conflict = True
                        break
                if conflict:
                    continue

            positions.append((px, py, pz))

        if len(positions) < n_target:
            print(f"  Warning: target={n_target}, placed={len(positions)}")
        return np.array(positions), attempts

    for n in [50, 100, 200, 500]:
        t0 = time.perf_counter()
        pos, attempts = generate_with_rejection(
            BOX_X, BOX_Y, SPACING, Z_SPREAD, n,
            min_dist=MIN_PAIR_DIST, surface_z=0.0, height_offset=SURFACE_HEIGHT_OFFSET,
        )
        dt = time.perf_counter() - t0
        acceptance = n / attempts * 100 if attempts > 0 else 0
        print(f"  N={n:4d}: {dt:.4f}s, attempts={attempts}, acceptance={acceptance:.1f}%")


def test_velocity():
    """测试速度方向 θ 正态 + φ 均匀"""
    print("\n" + "=" * 60)
    print("4. 速度方向测试")
    print("=" * 60)

    n_samples = 10000
    v_mag = 0.01
    theta_sigma_rad = math.radians(THETA_SIGMA_DEG)
    rng = np.random.default_rng(42)

    thetas = []
    phis = []
    for _ in range(n_samples):
        theta = abs(rng.normal(0, theta_sigma_rad))  # θ >= 0
        phi = rng.uniform(0, 2 * math.pi)
        thetas.append(theta)
        phis.append(phi)

        vx = v_mag * math.sin(theta) * math.cos(phi)
        vy = v_mag * math.sin(theta) * math.sin(phi)
        vz = -v_mag * math.cos(theta)

        # 验证模长
        v_actual = math.sqrt(vx*vx + vy*vy + vz*vz)
        if _ < 3:
            print(f"  Sample {_+1}: v=({vx:.5f}, {vy:.5f}, {vz:.5f}), |v|={v_actual:.5f}, θ={math.degrees(theta):.2f}°, φ={math.degrees(phi):.1f}°")

    thetas = np.array(thetas)
    phis = np.array(phis)
    print(f"  θ mean={np.degrees(np.mean(thetas)):.2f}°, std={np.degrees(np.std(thetas)):.2f}°")
    print(f"  φ 均匀性（0-2π 直方图 chi2）: ", end="")
    phi_hist, _ = np.histogram(phis, bins=20, range=[0, 2*math.pi])
    phi_expected = n_samples / 20
    phi_chi2 = np.sum((phi_hist - phi_expected)**2 / phi_expected)
    print(f"{phi_chi2:.1f}")


if __name__ == "__main__":
    np.random.seed(42)
    test_spatial_uniformity()
    test_pbc_randomness()
    test_min_distance_performance()
    test_velocity()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
