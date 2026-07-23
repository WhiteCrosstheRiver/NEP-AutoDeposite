import numpy as np
import random
import yaml
import os
import re
import shutil
import math
from collections import deque

# --- 物理常数 ---
KB = 1.380649e-23  # 玻尔兹曼常数 (J/K)
AMU = 1.660539e-27 # 原子质量单位 (kg)
M_TO_ANGSTROM = 1e10       # 米到埃的转换因子
S_TO_FS = 1e15              # 秒到飞秒的转换因子
M_S_TO_ANGSTROM_FS = M_TO_ANGSTROM / S_TO_FS  # m/s 转 Å/fs

def sample_mbe_velocity(element_symbol, temperature_k, direction_axis='z', rng=None):
    """
    生成更接近 MBE effusive beam 的入射速度
    - temperature_k: 建议使用蒸发源/Knudsen cell 温度
    - direction_axis: 束流主方向 ('x', 'y', 'z')
    返回单位: Å/fs
    """
    if rng is None:
        rng = np.random.default_rng()

    atomic_masses = {
        'Ge': 72.630,
        'Si': 28.085,
        'Ga': 69.723,
        'N': 14.007,
        'X': 69.723,
    }

    if element_symbol not in atomic_masses:
        raise ValueError(f"未知的元素符号: {element_symbol}")

    mass_kg = atomic_masses[element_symbol] * AMU

    # 热源中单分量的热速度标准差
    sigma = np.sqrt(KB * temperature_k / mass_kg)  # m/s

    # 横向分量：高斯
    v1 = rng.normal(0.0, sigma)
    v2 = rng.normal(0.0, sigma)

    # 法向分量：flux-weighted / effusive
    # p(v_n) = (v_n/sigma^2) * exp(-v_n^2/(2 sigma^2)), v_n > 0
    # 即 Rayleigh(scale=sigma)
    v_n = sigma * np.sqrt(-2.0 * np.log(rng.random()))  # m/s, > 0

    # 组装到指定方向，默认朝负方向入射
    vx = vy = vz = 0.0
    if direction_axis == 'z':
        vx, vy, vz = v1, v2, -v_n
    elif direction_axis == 'y':
        vx, vy, vz = v1, -v_n, v2
    elif direction_axis == 'x':
        vx, vy, vz = -v_n, v1, v2
    else:
        raise ValueError("direction_axis 必须是 'x', 'y' 或 'z'")

    # 单位转换: m/s -> Å/fs
    vx *= M_S_TO_ANGSTROM_FS
    vy *= M_S_TO_ANGSTROM_FS
    vz *= M_S_TO_ANGSTROM_FS

    return vx, vy, vz


def generate_maxwell_velocities(element_symbol, temperature_k, direction_axis='z'):
    """
    兼容旧接口：内部改用更接近 MBE effusive beam 的采样。
    """
    return sample_mbe_velocity(element_symbol, temperature_k, direction_axis=direction_axis)

def parse_substrate_replicate(value):
    """
    解析 total_config / config.yaml 中的 substrate_replicate。
    - 缺省 / None / 1 -> (1, 1)，不复制
    - 整数 N -> (N, N)
    - [nx, ny] -> (nx, ny)
    """
    if value is None:
        return 1, 1
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("substrate_replicate 列表需为 [nx, ny]")
        nx, ny = int(value[0]), int(value[1])
    else:
        nx = ny = int(value)
    if nx < 1 or ny < 1:
        raise ValueError("substrate_replicate 各分量须 >= 1")
    return nx, ny


def _parse_lattice_header(header_line):
    m = re.search(r'Lattice="([^"]*)"', header_line)
    if not m:
        raise ValueError("header 中未找到 Lattice")
    vals = list(map(float, m.group(1).split()))
    if len(vals) != 9:
        raise ValueError("Lattice 需要 9 个分量")
    return vals


def _replace_lattice_header(header_line, lattice_vals):
    new_lattice = " ".join(
        f"{v:.8g}" if abs(v) >= 1e-4 or v == 0 else f"{v:.8g}" for v in lattice_vals
    )

    def _sub(match):
        return f'Lattice="{new_lattice}"'

    return re.sub(r'Lattice="[^"]*"', _sub, header_line)


def _format_xyz_float(v):
    s = f"{float(v):.12f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def replicate_xy(input_path, output_path, nx=2, ny=2):
    """在 XY 平面做 nx×ny 复制，Z 与 box_z 不变；保持 extended xyz 格式。"""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    header = lines[1].rstrip("\n")
    atom_lines = [ln.rstrip("\n") for ln in lines[2:] if ln.strip()]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"{input_path}: 声明 {n_atoms} 原子，实际 {len(atom_lines)} 行")

    lattice = _parse_lattice_header(header)
    lx, ly = lattice[0], lattice[4]

    new_lattice = lattice[:]
    new_lattice[0] = lx * nx
    new_lattice[4] = ly * ny
    new_header = _replace_lattice_header(header, new_lattice)

    out_atoms = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"原子行格式错误: {line!r}")
        species = parts[0]
        x, y, z = map(float, parts[1:4])
        tail = parts[4:]
        for ix in range(nx):
            for iy in range(ny):
                nx_ = x + ix * lx
                ny_ = y + iy * ly
                cols = [species, _format_xyz_float(nx_), _format_xyz_float(ny_), _format_xyz_float(z)] + tail
                out_atoms.append(" ".join(cols))

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{len(out_atoms)}\n")
        f.write(new_header + "\n")
        for ln in out_atoms:
            f.write(ln + "\n")

    return {
        "input": input_path,
        "output": output_path,
        "n_in": n_atoms,
        "n_out": len(out_atoms),
        "box_in": (lx, ly, lattice[8]),
        "box_out": (new_lattice[0], new_lattice[4], new_lattice[8]),
    }


def prepare_initial_model_xyz(input_path, output_path, substrate_replicate=None):
    """
    首次迭代准备 model.xyz：可选 XY replicate，仅应在第一轮调用一次。
    """
    nx, ny = parse_substrate_replicate(substrate_replicate)
    if nx == 1 and ny == 1:
        shutil.copy2(input_path, output_path)
        return {"replicated": False, "nx": 1, "ny": 1}
    info = replicate_xy(input_path, output_path, nx=nx, ny=ny)
    info["replicated"] = True
    info["nx"] = nx
    info["ny"] = ny
    return info


def generate_run_in(output_dir, temperature, time_step=1, run_steps=10000, enable_d3=False):
    """
    生成 run.in 文件，温度作为可配置变量

    Parameters:
    -----------
    output_dir : str
        输出目录路径
    temperature : int or float
        目标温度 (K)
    time_step : int or float
        时间步长 (fs)，默认为1
    run_steps : int
        运行步数，默认为10000
    """
    d3_line = "dftd3 pbe 6 6\n" if enable_d3 else ""
    run_in_content = f"""potential   ../nep.txt

time_step   {time_step}

fix 1

dump_thermo 100
dump_exyz   {run_steps} 1 1
{d3_line}



# use a large tau_T to mimic NPH
ensemble    nvt_ber {temperature} {temperature} 10
run         {run_steps}

"""

    output_path = os.path.join(output_dir, 'run.in')
    with open(output_path, 'w') as f:
        f.write(run_in_content)

    print(f"Generated run.in in {output_dir} with temperature {temperature}K")

# 判断是否在某个特定 region
def is_in_region(position, region):
    """Check if a position is within specified region."""
    # region 是一个列表，包含两个元素：[min_coords, max_coords]
    min_coords, max_coords = region
    return all(min_coords[i] <= position[i] <= max_coords[i] for i in range(3))

def save_products_atom_list(products_atom_list, filename='products_atoms.yaml'):
    # 如果文件不存在，创建文件并初始化
    try:
        with open(filename, 'r') as file:
            data = yaml.safe_load(file) or []
    except FileNotFoundError:
        data = []

    # 将当前轮次的产物原子信息添加到数据列表中
    data.append({
        'round': len(data) + 1,
        'atoms': products_atom_list
    })

    # 将更新后的数据写回文件
    with open(filename, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def parse_properties(lattice_string):
    # 检查是否包含 "properties="（不区分大小写）
    if "properties=" not in lattice_string.lower():
        raise ValueError("The string does not contain 'properties='.")

    # 提取 properties= 后面的内容
    properties_start = lattice_string.lower().find("properties=")
    properties_part = lattice_string[properties_start + len("properties="):].split()[0]

    # 将内容分组，以 ':' 分割，并按每三项一组
    properties_groups = properties_part.split(':')

    # 检查是否可以按每三项分组
    if len(properties_groups) % 3 != 0:
        raise ValueError("The properties string is not formatted correctly.")

    # 将分组解析为简单的列表结构，每个元素是 [name, type, dimension]
    parsed_groups = []
    for i in range(0, len(properties_groups), 3):
        group = [
            properties_groups[i],
            properties_groups[i + 1],
            int(properties_groups[i + 2])
        ]
        parsed_groups.append(group)

    return parsed_groups


def _pbc_delta(a, b, box_len):
    delta = abs(a - b)
    if delta > box_len / 2.0:
        delta = box_len - delta
    return delta


def _find_main_component_from_min_z(positions, box_x, box_y, cutoff):
    """
    基于距离阈值构图，返回包含最低 z 原子的主连通集掩码。
    仅对 x/y 使用 PBC，z 方向不使用 PBC。
    """
    positions = np.asarray(positions, dtype=float)
    n_atoms = positions.shape[0]
    if n_atoms == 0:
        return np.asarray([], dtype=bool)
    if n_atoms == 1:
        return np.asarray([True], dtype=bool)

    cutoff = float(cutoff)
    cutoff_sq = cutoff * cutoff
    cell_size = max(cutoff, 1e-8)

    nx = max(1, int(np.ceil(box_x / cell_size)))
    ny = max(1, int(np.ceil(box_y / cell_size)))
    z_min = float(np.min(positions[:, 2]))
    z_max = float(np.max(positions[:, 2]))
    nz = max(1, int(np.ceil((z_max - z_min + 1e-8) / cell_size)))

    def xyz_to_cell(x, y, z):
        ix = int(np.floor(x / cell_size)) % nx
        iy = int(np.floor(y / cell_size)) % ny
        iz = int(np.floor((z - z_min) / cell_size))
        iz = max(0, min(nz - 1, iz))
        return ix, iy, iz

    cell_atoms = {}
    for idx, (x, y, z) in enumerate(positions):
        key = xyz_to_cell(x, y, z)
        cell_atoms.setdefault(key, []).append(idx)

    neighbors = [(-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
                 (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
                 (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
                 (0, -1, -1), (0, -1, 0), (0, -1, 1),
                 (0, 0, -1), (0, 0, 0), (0, 0, 1),
                 (0, 1, -1), (0, 1, 0), (0, 1, 1),
                 (1, -1, -1), (1, -1, 0), (1, -1, 1),
                 (1, 0, -1), (1, 0, 0), (1, 0, 1),
                 (1, 1, -1), (1, 1, 0), (1, 1, 1)]

    seed = int(np.argmin(positions[:, 2]))
    visited = np.zeros(n_atoms, dtype=bool)
    visited[seed] = True
    q = deque([seed])

    while q:
        i = q.popleft()
        xi, yi, zi = positions[i]
        cix, ciy, ciz = xyz_to_cell(xi, yi, zi)

        for ox, oy, oz in neighbors:
            nix = (cix + ox) % nx
            niy = (ciy + oy) % ny
            niz = ciz + oz
            if niz < 0 or niz >= nz:
                continue
            for j in cell_atoms.get((nix, niy, niz), []):
                if visited[j] or j == i:
                    continue
                xj, yj, zj = positions[j]
                dx = _pbc_delta(xi, xj, box_x)
                dy = _pbc_delta(yi, yj, box_y)
                dz = zi - zj
                if dx * dx + dy * dy + dz * dz <= cutoff_sq:
                    visited[j] = True
                    q.append(j)

    return visited


def filter_main_deposit(atoms_list, box_x, box_y, cluster_cutoff=2.8):
    """
    保留与最低 z 原子连通的主沉积体，剔除孤立入射/飞溅粒子。

    Returns
    -------
    filtered_atoms : list
    n_removed : int
    deposited_positions : ndarray, shape (n_keep, 3)
    """
    if not atoms_list:
        return [], 0, np.empty((0, 3), dtype=float)

    positions = np.asarray([atom['pos'] for atom in atoms_list], dtype=float)
    keep_mask = _find_main_component_from_min_z(
        positions, box_x, box_y, cluster_cutoff
    )
    filtered_atoms = [atom for i, atom in enumerate(atoms_list) if keep_mask[i]]
    n_removed = int((~keep_mask).sum())
    deposited_positions = positions[keep_mask]
    return filtered_atoms, n_removed, deposited_positions


def get_local_surface_z_from_atoms(
    positions,
    px,
    py,
    box_x,
    box_y,
    local_search_radius=5.0,
    cluster_cutoff=2.8,
    deposited_positions=None,
):
    """
    利用主连通集过滤后，返回 (px, py) 局部最高表面 z。
    若传入 deposited_positions（已过滤的主沉积体坐标），则跳过连通集计算。
    """
    if deposited_positions is not None:
        deposited_atoms = np.asarray(deposited_positions, dtype=float)
    else:
        positions = np.asarray(positions, dtype=float)
        if positions.size == 0:
            return 0.0
        keep_mask = _find_main_component_from_min_z(
            positions, box_x, box_y, cluster_cutoff
        )
        deposited_atoms = positions[keep_mask]

    if deposited_atoms.size == 0:
        return 0.0

    dx = np.abs(deposited_atoms[:, 0] - px)
    dy = np.abs(deposited_atoms[:, 1] - py)
    dx = np.where(dx > box_x / 2.0, box_x - dx, dx)
    dy = np.where(dy > box_y / 2.0, box_y - dy, dy)
    dist_sq = dx**2 + dy**2

    local_mask = dist_sq < (local_search_radius ** 2)
    local_atoms = deposited_atoms[local_mask]
    if local_atoms.shape[0] > 0:
        return float(np.max(local_atoms[:, 2]))
    return float(np.max(deposited_atoms[:, 2]))


def injection_count_from_spacing(box_x, box_y, spacing):
    """按 XY 间距向下取整格数，返回 (N, nx, ny)。"""
    spacing = float(spacing)
    if spacing <= 0:
        raise ValueError("inject_xy_spacing 必须 > 0")
    nx = int(np.floor(float(box_x) / spacing))
    ny = int(np.floor(float(box_y) / spacing))
    return nx * ny, nx, ny


def generate_gaussian_beam_velocity(velocity_magnitude, theta_sigma_deg=5.0, rng=None):
    """
    腔体束流速度：|v| 固定，θ ~ |Normal(0, σ_θ)|，φ ~ Uniform(0, 2π)。
    高斯仅影响速度方向（束流发散），不影响粒子 XY 位置。
    返回单位: Å/fs
    """
    if rng is None:
        rng = np.random.default_rng()
    theta_sigma_rad = math.radians(float(theta_sigma_deg))
    theta = abs(float(rng.normal(0.0, theta_sigma_rad)))
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    v_mag = float(velocity_magnitude)
    vx = v_mag * math.sin(theta) * math.cos(phi)
    vy = v_mag * math.sin(theta) * math.sin(phi)
    vz = -v_mag * math.cos(theta)
    return vx, vy, vz


def generate_cavity_particle_positions(
    base_positions,
    box_x,
    box_y,
    inject_xy_spacing,
    n_target,
    z_spread,
    local_search_radius=5.0,
    surface_height_offset=15.0,
    cluster_cutoff=2.8,
    deposited_positions=None,
    rng=None,
):
    """
    腔体粒子位置：细网格 + 每轮随机 offset 的 XY；Z 在局部表面 + offset 至 +z_spread 均匀分布。
    """
    if rng is None:
        rng = np.random.default_rng()
    spacing = float(inject_xy_spacing)
    nx = int(np.floor(float(box_x) / spacing))
    ny = int(np.floor(float(box_y) / spacing))
    if nx <= 0 or ny <= 0:
        return []

    offset_x = float(rng.uniform(0, spacing))
    offset_y = float(rng.uniform(0, spacing))
    xs = (np.arange(nx, dtype=float) + 0.5) * spacing + offset_x
    ys = (np.arange(ny, dtype=float) + 0.5) * spacing + offset_y
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
    n_grid = len(grid_xy)

    perm = rng.permutation(n_grid)
    grid_xy = grid_xy[perm]

    n_target = max(0, int(n_target))
    if n_target <= n_grid:
        xy = grid_xy[:n_target]
    else:
        extra = n_target - n_grid
        xy_extra = np.column_stack([
            rng.uniform(0, box_x, size=extra),
            rng.uniform(0, box_y, size=extra),
        ])
        xy = np.vstack([grid_xy, xy_extra])

    z_spread = float(z_spread)
    positions = []
    for px, py in xy:
        px = float(np.mod(px, box_x))
        py = float(np.mod(py, box_y))
        z_surf = get_local_surface_z_from_atoms(
            base_positions,
            px,
            py,
            box_x,
            box_y,
            local_search_radius=local_search_radius,
            cluster_cutoff=cluster_cutoff,
            deposited_positions=deposited_positions,
        )
        base_z = z_surf + float(surface_height_offset)
        pz = float(rng.uniform(base_z, base_z + z_spread))
        positions.append((px, py, pz))
    return positions


def generate_grid_particle_positions(
    base_positions,
    box_x,
    box_y,
    inject_xy_spacing,
    local_search_radius=5.0,
    surface_height_offset=15.0,
    cluster_cutoff=2.8,
    deposited_positions=None,
    z_spread=0.0,
    injection_count=None,
):
    """兼容旧函数名：内部调用腔体网格 XY + 局部表面 Z + z_spread。"""
    if injection_count is not None:
        n_target = max(0, int(injection_count))
    else:
        n_target, _, _ = injection_count_from_spacing(box_x, box_y, inject_xy_spacing)
    return generate_cavity_particle_positions(
        base_positions=base_positions,
        box_x=box_x,
        box_y=box_y,
        inject_xy_spacing=inject_xy_spacing,
        n_target=n_target,
        z_spread=z_spread,
        local_search_radius=local_search_radius,
        surface_height_offset=surface_height_offset,
        cluster_cutoff=cluster_cutoff,
        deposited_positions=deposited_positions,
    )


def generate_particle_positions(
    base_positions,
    box_x,
    box_y,
    inject_xy_spacing,
    local_search_radius=5.0,
    surface_height_offset=15.0,
    cluster_cutoff=2.8,
    deposited_positions=None,
    z_spread=0.0,
    injection_count=None,
):
    """兼容旧函数名：内部调用腔体网格方案。"""
    return generate_grid_particle_positions(
        base_positions=base_positions,
        box_x=box_x,
        box_y=box_y,
        inject_xy_spacing=inject_xy_spacing,
        local_search_radius=local_search_radius,
        surface_height_offset=surface_height_offset,
        cluster_cutoff=cluster_cutoff,
        deposited_positions=deposited_positions,
        z_spread=z_spread,
        injection_count=injection_count,
    )

def last_converted_new(
    file_path='dump.xyz',
    output_file='test.xyz',
    inject_temperature=None,
    injection_flux=1e-7,
    box_z=1000.0,
    time_step=1,
    run_steps=1000,
    inject_species_weights=None,
    inject_xy_spacing=8.0,
    injection_count=None,
    z_spread=50.0,
    local_surface_radius=5.0,
    surface_height_offset=15.0,
    cluster_cutoff=2.8,
    remove_incident_particles=True,
    velocity_magnitude=None,
    theta_sigma_deg=5.0,
):
    """
    处理xyz文件，添加入射粒子

    Parameters:
    -----------
    file_path : str
        输入文件路径
    output_file : str
        输出文件路径
    inject_temperature : float or None
        入射粒子温度 (K)
    injection_flux : float
        入射粒子横截面流速 (原子/(Å²·fs))
    remove_incident_particles : bool
        为 True 时，在注入前剔除与主沉积体不连通的孤立入射粒子
    inject_xy_spacing : float or None
        XY 网格间距（Å）；腔体方案下决定网格密度
    z_spread : float
        腔体 Z 均匀分布宽度（Å），相对局部表面 + offset
    velocity_magnitude : float or None
        入射速度模长（Å/fs）；若与 inject_temperature 同时设置，优先使用此项
    theta_sigma_deg : float
        束流发散角 σ_θ（度），θ ~ |Normal(0, σ_θ)|，仅影响速度方向
    """
    # 从lattice行中提取box参数
    with open(file_path, 'r') as f:
        first_lines = f.readlines()[:2]
    lattice_str = first_lines[1]
    # 解析box参数
    box_match = re.search(r'Lattice="([^"]*)"', lattice_str)
    if box_match:
        box_params = list(map(float, box_match.group(1).split()))
        box_x, box_y = box_params[0], box_params[4]
    else:
        print(f"Warning: Failed to parse Lattice string: {lattice_str.strip()}")
        box_x, box_y = 32.6622, 65.3244  # 默认值


    CONFINED_REGION_SI_BOTTOM_LIST=[[-2.0,-2.0,0.0],[box_x,box_y,3.0]]
    CONFINED_REGION=[CONFINED_REGION_SI_BOTTOM_LIST]  #这个区域固定哪些原子需要FIX
    RESERVED_REGION=[[-2.0,-2.0,-2.0],[box_x+2.0,box_y+2.0,box_z-50.0]]   #这个区域的原子保留

    with open(file_path, 'r') as file:
        lines = file.readlines()
    total_nums=int(lines[0])
    lattice_label=parse_properties(lines[1])

    info_data=lines[1]

    # 读取前xyz文件为 atoms_list
    atoms_list = []
    for line in lines[2:]:
        parts = line.split()
        atom={}
        idx_num=0
        for label_idx in range(len(lattice_label)):
            label=lattice_label[label_idx]

            if label[0] in ['species']:
                atom[label[0]]=parts[idx_num]
            if label[0] in ['pos','vel','forces']:
                atom[label[0]]=list(map(float, parts[idx_num:idx_num+3]))
            if label[0] in ['charge','mass']:
                atom[label[0]]=float(parts[idx_num])
            if label[0] in ['group']:
                atom[label[0]]=list(map(int, parts[idx_num:idx_num+3]))
            idx_num+=label[2]
        atoms_list.append(atom)

    deposited_positions = None
    if remove_incident_particles:
        atoms_list, n_removed, deposited_positions = filter_main_deposit(
            atoms_list, box_x, box_y, cluster_cutoff=float(cluster_cutoff)
        )
        if n_removed:
            print(
                f"Removed {n_removed} disconnected incident particle(s) "
                f"(cluster_cutoff={cluster_cutoff})"
            )

    # 判断是否特定区域需要标记 group 0 1
    # 确定 group 长度
    if atoms_list and 'group' in atoms_list[0]:
        group_length = len(atoms_list[0]['group'])
    else:
        group_length = 1

    for atom in atoms_list:
        if any(is_in_region(atom['pos'], region) for region in CONFINED_REGION):
            atom['group']=[1] + [0]*(group_length-1)
        else:
            atom['group']=[0]*group_length

    base_positions = np.asarray([atom['pos'] for atom in atoms_list], dtype=float)
    if deposited_positions is None:
        deposited_positions = base_positions

    # 腔体方案：网格 XY + 局部表面 Z + z_spread
    if inject_xy_spacing is not None:
        if injection_count is not None:
            num_atoms_to_add = max(0, int(injection_count))
            grid_nx = int(np.floor(float(box_x) / float(inject_xy_spacing)))
            grid_ny = int(np.floor(float(box_y) / float(inject_xy_spacing)))
        else:
            num_atoms_to_add, grid_nx, grid_ny = injection_count_from_spacing(
                box_x, box_y, inject_xy_spacing
            )
        candidate_positions = generate_cavity_particle_positions(
            base_positions=base_positions,
            box_x=box_x,
            box_y=box_y,
            inject_xy_spacing=float(inject_xy_spacing),
            n_target=num_atoms_to_add,
            z_spread=float(z_spread),
            local_search_radius=float(local_surface_radius),
            surface_height_offset=float(surface_height_offset),
            cluster_cutoff=float(cluster_cutoff),
            deposited_positions=deposited_positions,
        )
        print(
            f"Number of atoms to add: {num_atoms_to_add} "
            f"(cavity grid spacing={inject_xy_spacing}, {grid_nx}x{grid_ny}, "
            f"z_spread={z_spread})"
        )
    elif injection_count is not None:
        num_atoms_to_add = max(0, int(injection_count))
        print(f"Number of atoms to add: {num_atoms_to_add} (fixed injection_count, random XY)")
        candidate_positions = []
        rng = np.random.default_rng()
        for _ in range(num_atoms_to_add):
            px = float(rng.uniform(0.0, box_x))
            py = float(rng.uniform(0.0, box_y))
            z_surf = get_local_surface_z_from_atoms(
                base_positions,
                px,
                py,
                box_x,
                box_y,
                local_search_radius=float(local_surface_radius),
                cluster_cutoff=float(cluster_cutoff),
                deposited_positions=deposited_positions,
            )
            base_z = z_surf + float(surface_height_offset)
            pz = float(rng.uniform(base_z, base_z + float(z_spread)))
            candidate_positions.append((px, py, pz))
    else:
        lambda_poisson = injection_flux * box_x * box_y * time_step * run_steps
        num_atoms_to_add = int(np.random.poisson(lambda_poisson))
        print(f"Number of atoms to add: {num_atoms_to_add} (lambda={lambda_poisson:.2f}, random XY)")
        candidate_positions = []
        rng = np.random.default_rng()
        for _ in range(num_atoms_to_add):
            px = float(rng.uniform(0.0, box_x))
            py = float(rng.uniform(0.0, box_y))
            z_surf = get_local_surface_z_from_atoms(
                base_positions,
                px,
                py,
                box_x,
                box_y,
                local_search_radius=float(local_surface_radius),
                cluster_cutoff=float(cluster_cutoff),
                deposited_positions=deposited_positions,
            )
            candidate_positions.append((px, py, z_surf + surface_height_offset))

    if len(candidate_positions) < num_atoms_to_add:
        print(
            f"Warning: target={num_atoms_to_add}, placed={len(candidate_positions)} "
            f"(inject_xy_spacing={inject_xy_spacing})"
        )

    # 添加多个原子
    for px, py, pz in candidate_positions:
            # 从配置文件读取物种权重
            if inject_species_weights is None:
                # 默认值：Ge (30%) 或 Si (70%)
                inject_species_weights = {'Ge': 0.3, 'Si': 0.7}
            
            species_list = list(inject_species_weights.keys())
            weights_list = list(inject_species_weights.values())
            species_inject = random.choices(species_list, weights=weights_list)[0]

            # 生成速度：velocity_magnitude 优先，否则回退 MBE 温度采样
            if velocity_magnitude is not None:
                vx, vy, vz = generate_gaussian_beam_velocity(
                    velocity_magnitude,
                    theta_sigma_deg=theta_sigma_deg,
                )
                vel = [vx, vy, vz]
            elif inject_temperature is not None:
                vx, vy, vz = sample_mbe_velocity(species_inject, inject_temperature, direction_axis='z')
                vel = [vx, vy, vz]
            else:
                # 使用默认固定速度
                vel = [0, 0, -0.001]

            atom_inject = {
                'species': species_inject,
                'pos': [px, py, pz],
                'vel': vel,
                'group': [0]*group_length
            }
            atoms_list.append(atom_inject)

    products_atom_list=[]
    #这些区域的原子保留，超出的全部删除
    # RESERVED_REGION 已经是一个完整的 region 定义（包含 min 和 max 坐标）
    atoms_list = [atom for atom in atoms_list if is_in_region(atom['pos'], RESERVED_REGION)]

    #输出xyz
    with open(output_file, 'w') as file:
        file.write(str(len(atoms_list))+'\n')

        # 先替换原始 info_data 中的 Lattice Z 值
        def replace_lattice_z(m):
            lattice_values = m.group(1).split()
            lattice_values[-1] = f"{box_z:.8f}"
            return f'Lattice="{" ".join(lattice_values)}"'

        updated_info_data = re.sub(r'Lattice="([^"]*)"', replace_lattice_z, info_data)

        # 然后处理 Properties 部分
        info_data_list = updated_info_data.split()
        info_data_list.pop(-1)  # 移除末尾的换行符

        # 检查是否有原子，如果有则检查属性
        has_vel = False
        has_group = False
        if atoms_list:
            has_vel = 'vel' in atoms_list[0]
            has_group = 'group' in atoms_list[0]

        if has_vel:
            properties_data='Properties=species:S:1:pos:R:3:vel:R:3\n'
        else:
            properties_data='Properties=species:S:1:pos:R:3\n'
        info_data_list.append(properties_data)

        final_info_data = ' '.join(info_data_list)
        if has_group:
            file.write(final_info_data.rstrip('\n')+':group:I:'+str(group_length)+'\n')
        else:
            file.write(final_info_data)

        for atom in atoms_list:
            # 使用格式化字符串来格式化输出
            pos_str = ' '.join(f"{p:8.3f}" for p in atom['pos'])  # 格式化坐标，保留三位小数
            if  'vel' in atom:
                vel_str = ' '.join(f"{p:8.3f}" for p in atom['vel'])  # 格式化坐标，保留三位小数
            group_str = ' '.join(f"{d:4d}" for d in atom['group'])  # 格式化其他数据为整数，保留宽度为4
            if  'vel' in atom:
                line = f"{atom['species']:<2s} {pos_str} {vel_str} {group_str}\n"  # 对元素名称进行右对齐
            else:
                line = f"{atom['species']:<2s} {pos_str} {group_str}\n"  # 对元素名称进行右对齐
            file.write(line)

    return products_atom_list
