"""SoA array pipeline: parse dump -> filter -> inject -> write model."""

import io
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import random

from surface_grid import SurfaceGrid2D
from utility import (
    _pbc_wrap_xy,
    _resolve_intra_round_overlap,
    format_bfs_light_log,
    generate_gaussian_beam_velocity,
    load_fixed_inject_xy,
    resolve_bfs_keep_mask,
)

try:
    from inject_fast import (
        _HAS_NUMBA_INJECT,
        query_zmax_batch,
        resolve_overlap_batch,
    )
except ImportError:
    _HAS_NUMBA_INJECT = False
    query_zmax_batch = None
    resolve_overlap_batch = None

try:
    import fast_xyz as _fast_xyz

    _HAS_FAST_XYZ = True
except ImportError:
    _fast_xyz = None
    _HAS_FAST_XYZ = False

SPECIES_TO_INT = {"Si": 0, "Ge": 1}
INT_TO_SPECIES = np.array(["Si", "Ge"], dtype=object)


@dataclass
class AtomFrame:
    species: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    group: np.ndarray

    @property
    def n(self):
        return self.species.shape[0]

    def slice_mask(self, mask):
        return AtomFrame(
            species=self.species[mask],
            pos=self.pos[mask],
            vel=self.vel[mask],
            group=self.group[mask],
        )

    def append(self, other):
        return AtomFrame(
            species=np.concatenate([self.species, other.species]),
            pos=np.vstack([self.pos, other.pos]),
            vel=np.vstack([self.vel, other.vel]),
            group=np.concatenate([self.group, other.group]),
        )


@dataclass
class PipelineTiming:
    parse_ms: float = 0.0
    bfs_ms: float = 0.0
    surface_ms: float = 0.0
    inject_ms: float = 0.0
    write_ms: float = 0.0

    @property
    def total_ms(self):
        return self.parse_ms + self.bfs_ms + self.surface_ms + self.inject_ms + self.write_ms


def _parse_box_xy(header_line):
    box_match = re.search(r'Lattice="([^"]*)"', header_line)
    if not box_match:
        raise ValueError("header 中未找到 Lattice")
    box_params = list(map(float, box_match.group(1).split()))
    return float(box_params[0]), float(box_params[4])


def _build_model_header(header_line, box_z):
    def replace_lattice_z(m):
        lattice_values = m.group(1).split()
        lattice_values[-1] = f"{float(box_z):.8f}"
        return f'Lattice="{" ".join(lattice_values)}"'

    updated = re.sub(r'Lattice="([^"]*)"', replace_lattice_z, header_line)
    info_parts = updated.split()
    if info_parts and info_parts[-1].startswith("Properties="):
        info_parts.pop()
    info_parts.append("Properties=species:S:1:pos:R:3:vel:R:3:group:I:1")
    return " ".join(info_parts)


def parse_dump_soa(path, fast_io=True, parser_mode="openmp"):
    """Parse extended xyz dump; read species + pos + vel only (skip forces)."""
    if fast_io and _HAS_FAST_XYZ:
        species, pos, vel, header = _fast_xyz.read_dump_xyz(
            path, skip_forces=True, parser_mode=str(parser_mode)
        )
        species = np.asarray(species, dtype=np.int8)
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        vel = np.ascontiguousarray(vel, dtype=np.float64)
        header = header.rstrip("\n")
        box_x, box_y = _parse_box_xy(header + "\n")
        group = np.zeros(species.shape[0], dtype=np.uint8)
        frame = AtomFrame(species=species, pos=pos, vel=vel, group=group)
        return frame, box_x, box_y, header

    with open(path, "r", encoding="utf-8") as f:
        n_atoms = int(f.readline().strip())
        header = f.readline()
        box_x, box_y = _parse_box_xy(header)

        species = np.empty(n_atoms, dtype=np.int8)
        numeric_lines = []
        for i in range(n_atoms):
            parts = f.readline().split()
            sp = parts[0]
            if sp not in SPECIES_TO_INT:
                raise ValueError(f"未知 species: {sp!r}")
            species[i] = SPECIES_TO_INT[sp]
            numeric_lines.append(
                f"{parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]} {parts[6]}"
            )

    nums = np.loadtxt(
        io.StringIO("\n".join(numeric_lines)),
        dtype=np.float64,
    )
    if nums.ndim == 1:
        nums = nums.reshape(1, 6)
    pos = nums[:, 0:3]
    vel = nums[:, 3:6]
    group = np.zeros(n_atoms, dtype=np.uint8)
    frame = AtomFrame(species=species, pos=pos, vel=vel, group=group)
    return frame, box_x, box_y, header.rstrip("\n")


def filter_main_deposit_soa(
    frame,
    box_x,
    box_y,
    cluster_cutoff,
    bfs_mode="light",
    bfs_full_interval=10,
    bfs_light_percentile=90.0,
    round_num=1,
):
    """Keep main connected component; return (filtered_frame, n_removed, bfs_stats)."""
    keep_mask, stats = resolve_bfs_keep_mask(
        frame.pos,
        box_x,
        box_y,
        float(cluster_cutoff),
        bfs_mode=bfs_mode,
        bfs_full_interval=bfs_full_interval,
        bfs_light_percentile=bfs_light_percentile,
        round_num=round_num,
    )
    n_removed = int((~keep_mask).sum())
    if stats.mode != "shadow":
        stats.n_removed = n_removed
    else:
        stats.full_removed = n_removed
    return frame.slice_mask(keep_mask), n_removed, stats


def apply_group_labels(frame, box_x, box_y):
    """Vectorized fix group for substrate bottom z in [0, 3]."""
    confined = (
        (frame.pos[:, 0] >= -2.0)
        & (frame.pos[:, 0] <= box_x)
        & (frame.pos[:, 1] >= -2.0)
        & (frame.pos[:, 1] <= box_y)
        & (frame.pos[:, 2] >= 0.0)
        & (frame.pos[:, 2] <= 3.0)
    )
    frame.group[:] = 0
    frame.group[confined] = 1
    return frame


def crop_reserved_region(frame, box_x, box_y, box_z):
    """Keep atoms inside RESERVED_REGION."""
    z_max = float(box_z) - 50.0
    mask = (
        (frame.pos[:, 0] >= -2.0)
        & (frame.pos[:, 0] <= box_x + 2.0)
        & (frame.pos[:, 1] >= -2.0)
        & (frame.pos[:, 1] <= box_y + 2.0)
        & (frame.pos[:, 2] >= -2.0)
        & (frame.pos[:, 2] <= z_max)
    )
    return frame.slice_mask(mask)


def generate_injections_soa_legacy(
    fixed_xy_points,
    surface_grid,
    box_x,
    box_y,
    local_search_radius,
    velocity_magnitude,
    theta_sigma_deg,
    placement_bond_cutoff,
    placement_distance_factor,
    inject_xy_gaussian_3sigma,
    inject_species_weights,
    rng,
):
    """Original per-particle Python loop (kept for comparison)."""
    if inject_species_weights is None:
        inject_species_weights = {"Ge": 1.0, "Si": 0.0}
    species_list = list(inject_species_weights.keys())
    weights_list = list(inject_species_weights.values())

    sep = float(placement_distance_factor) * float(placement_bond_cutoff)
    xy_sigma = float(inject_xy_gaussian_3sigma) / 3.0
    min_intra_sep = max(2.8, sep * 0.85)
    placed_positions = []

    n_inj = len(fixed_xy_points)
    inj_species = np.empty(n_inj, dtype=np.int8)
    inj_pos = np.empty((n_inj, 3), dtype=np.float64)
    inj_vel = np.empty((n_inj, 3), dtype=np.float64)

    for i, (px0, py0) in enumerate(fixed_xy_points):
        vx, vy, vz = generate_gaussian_beam_velocity(
            velocity_magnitude,
            theta_sigma_deg=theta_sigma_deg,
            rng=rng,
        )
        sp_name = random.choices(species_list, weights=weights_list)[0]
        inj_species[i] = SPECIES_TO_INT.get(sp_name, 1)

        px = float(px0) + float(rng.normal(0.0, xy_sigma))
        py = float(py0) + float(rng.normal(0.0, xy_sigma))
        px, py = _pbc_wrap_xy(px, py, box_x, box_y)

        zmax_atom = surface_grid.query_zmax_atom(
            px, py, radius=float(local_search_radius)
        )
        pz = float(zmax_atom[2] + sep)

        pos = _resolve_intra_round_overlap(
            [px, py, pz],
            [vx, vy, vz],
            placed_positions,
            box_x,
            box_y,
            min_intra_sep,
        )
        placed_positions.append(np.array(pos, dtype=np.float64))
        inj_pos[i] = pos
        inj_vel[i, 0] = vx
        inj_vel[i, 1] = vy
        inj_vel[i, 2] = vz

    inj_group = np.zeros(n_inj, dtype=np.uint8)
    return AtomFrame(
        species=inj_species, pos=inj_pos, vel=inj_vel, group=inj_group
    ), sep, xy_sigma


def generate_injections_soa(
    fixed_xy_points,
    surface_grid,
    box_x,
    box_y,
    local_search_radius,
    velocity_magnitude,
    theta_sigma_deg,
    placement_bond_cutoff,
    placement_distance_factor,
    inject_xy_gaussian_3sigma,
    inject_species_weights,
    rng,
    inject_fast=True,
):
    """Array-based gaussian-spot near-surface injection (same physics as v5)."""
    if not inject_fast or not _HAS_NUMBA_INJECT:
        return generate_injections_soa_legacy(
            fixed_xy_points,
            surface_grid,
            box_x,
            box_y,
            local_search_radius,
            velocity_magnitude,
            theta_sigma_deg,
            placement_bond_cutoff,
            placement_distance_factor,
            inject_xy_gaussian_3sigma,
            inject_species_weights,
            rng,
        )

    if inject_species_weights is None:
        inject_species_weights = {"Ge": 1.0, "Si": 0.0}

    sep = float(placement_distance_factor) * float(placement_bond_cutoff)
    xy_sigma = float(inject_xy_gaussian_3sigma) / 3.0
    min_intra_sep = max(2.8, sep * 0.85)
    n_inj = len(fixed_xy_points)

    inj_species = np.empty(n_inj, dtype=np.int8)
    inj_vel = np.empty((n_inj, 3), dtype=np.float64)
    px_arr = np.empty(n_inj, dtype=np.float64)
    py_arr = np.empty(n_inj, dtype=np.float64)

    if inject_species_weights is None:
        inject_species_weights = {"Ge": 1.0, "Si": 0.0}
    species_list = list(inject_species_weights.keys())
    weights_list = list(inject_species_weights.values())

    # RNG 顺序与 legacy 完全一致（逐粒子），仅几何阶段 batch 化
    for i, (px0, py0) in enumerate(fixed_xy_points):
        vx, vy, vz = generate_gaussian_beam_velocity(
            velocity_magnitude,
            theta_sigma_deg=theta_sigma_deg,
            rng=rng,
        )
        sp_name = random.choices(species_list, weights=weights_list)[0]
        inj_species[i] = SPECIES_TO_INT.get(sp_name, 1)
        inj_vel[i, 0] = vx
        inj_vel[i, 1] = vy
        inj_vel[i, 2] = vz
        px, py = _pbc_wrap_xy(
            float(px0) + float(rng.normal(0.0, xy_sigma)),
            float(py0) + float(rng.normal(0.0, xy_sigma)),
            box_x,
            box_y,
        )
        px_arr[i] = px
        py_arr[i] = py

    zmax_atoms = query_zmax_batch(
        surface_grid, px_arr, py_arr, float(local_search_radius)
    )
    init_pos = np.empty((n_inj, 3), dtype=np.float64)
    init_pos[:, 0] = px_arr
    init_pos[:, 1] = py_arr
    init_pos[:, 2] = zmax_atoms[:, 2] + sep

    inj_pos = resolve_overlap_batch(
        init_pos, inj_vel, float(box_x), float(box_y), min_intra_sep
    )

    inj_group = np.zeros(n_inj, dtype=np.uint8)
    return AtomFrame(
        species=inj_species, pos=inj_pos, vel=inj_vel, group=inj_group
    ), sep, xy_sigma


def write_model_xyz(
    frame,
    output_path,
    header_line,
    box_z,
    fast_io=True,
    xyz_float_decimals=3,
):
    """Write model.xyz; use C++ fast_xyz when available."""
    np.nan_to_num(frame.pos, copy=False, nan=0.0)
    np.nan_to_num(frame.vel, copy=False, nan=0.0)
    header = _build_model_header(header_line, box_z)

    if fast_io and _HAS_FAST_XYZ:
        decimals = int(xyz_float_decimals)
        if decimals not in (3, 8):
            decimals = 3
        _fast_xyz.write_model_xyz(
            output_path,
            np.ascontiguousarray(frame.species, dtype=np.uint8),
            np.ascontiguousarray(frame.pos, dtype=np.float64),
            np.ascontiguousarray(frame.vel, dtype=np.float64),
            np.ascontiguousarray(frame.group, dtype=np.uint8),
            header,
            decimals,
        )
        return

    n = frame.n
    sp_names = INT_TO_SPECIES[frame.species]
    pos = frame.pos
    vel = frame.vel
    grp = frame.group.astype(np.int64)

    with open(output_path, "w", encoding="utf-8", buffering=16 * 1024 * 1024) as f:
        f.write(f"{n}\n{header}\n")
        chunk = 8192
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            lines = [
                (
                    f"{sp_names[i]:<2s} "
                    f"{pos[i, 0]:8.3f} {pos[i, 1]:8.3f} {pos[i, 2]:8.3f} "
                    f"{vel[i, 0]:8.3f} {vel[i, 1]:8.3f} {vel[i, 2]:8.3f} "
                    f"{grp[i]:4d}\n"
                )
                for i in range(start, end)
            ]
            f.write("".join(lines))


def last_converted_new_soa(
    file_path,
    output_file,
    box_z=200.0,
    inject_species_weights=None,
    local_surface_radius=5.0,
    cluster_cutoff=3.2,
    remove_incident_particles=True,
    bfs_mode="light",
    bfs_full_interval=10,
    bfs_light_percentile=90.0,
    round_num=1,
    velocity_magnitude=0.01,
    theta_sigma_deg=5.0,
    inject_mode=None,
    fixed_inject_xy_path=None,
    placement_bond_cutoff=4.0,
    placement_distance_factor=0.65,
    inject_xy_gaussian_3sigma=20.0,
    surface_grid_cell_size=5.0,
    inject_round_seed=None,
    pipeline_timeit=False,
    fast_io=True,
    xyz_float_decimals=3,
    parser_mode="openmp",
    inject_fast=True,
    **_ignored,
):
    """
    SoA replacement for v5 last_converted_new (fast_ballistic_near_surface mode).
    Returns PipelineTiming.
    """
    timing = PipelineTiming()

    t0 = time.perf_counter()
    frame, box_x, box_y, header = parse_dump_soa(
        file_path, fast_io=fast_io, parser_mode=parser_mode
    )
    timing.parse_ms = (time.perf_counter() - t0) * 1000.0

    n_removed = 0
    bfs_stats = None
    if remove_incident_particles:
        t0 = time.perf_counter()
        frame, n_removed, bfs_stats = filter_main_deposit_soa(
            frame,
            box_x,
            box_y,
            cluster_cutoff,
            bfs_mode=bfs_mode,
            bfs_full_interval=bfs_full_interval,
            bfs_light_percentile=bfs_light_percentile,
            round_num=round_num,
        )
        timing.bfs_ms = (time.perf_counter() - t0) * 1000.0
        if n_removed:
            print(
                f"Removed {n_removed} disconnected incident particle(s) "
                f"(cluster_cutoff={cluster_cutoff}, bfs_mode={bfs_mode})"
            )
        if bfs_stats is not None and bfs_mode in ("light", "shadow"):
            print(format_bfs_light_log(bfs_stats, round_num, frame.n + n_removed))

    frame = apply_group_labels(frame, box_x, box_y)
    dep_pos = frame.pos

    if inject_mode != "fast_ballistic_near_surface":
        raise ValueError(f"v6 SoA pipeline 仅支持 fast_ballistic_near_surface，got {inject_mode!r}")

    if not fixed_inject_xy_path or not os.path.isfile(fixed_inject_xy_path):
        raise ValueError(
            f"fast_ballistic_near_surface 需要 fixed_inject_xy_path (got {fixed_inject_xy_path!r})"
        )

    fixed_xy_points, _meta = load_fixed_inject_xy(fixed_inject_xy_path)
    num_atoms_to_add = len(fixed_xy_points)

    t0 = time.perf_counter()
    surface_grid = SurfaceGrid2D(
        dep_pos,
        box_x,
        box_y,
        cell_size=float(surface_grid_cell_size or local_surface_radius),
    )
    timing.surface_ms = (time.perf_counter() - t0) * 1000.0

    if inject_round_seed is not None:
        rng = np.random.default_rng(int(inject_round_seed))
        random.seed(int(inject_round_seed))
    else:
        rng = np.random.default_rng()

    t0 = time.perf_counter()
    inj_frame, sep, xy_sigma = generate_injections_soa(
        fixed_xy_points=fixed_xy_points,
        surface_grid=surface_grid,
        box_x=box_x,
        box_y=box_y,
        local_search_radius=float(local_surface_radius),
        velocity_magnitude=float(velocity_magnitude),
        theta_sigma_deg=float(theta_sigma_deg),
        placement_bond_cutoff=float(placement_bond_cutoff),
        placement_distance_factor=float(placement_distance_factor),
        inject_xy_gaussian_3sigma=float(inject_xy_gaussian_3sigma),
        inject_species_weights=inject_species_weights,
        rng=rng,
        inject_fast=inject_fast,
    )
    timing.inject_ms = (time.perf_counter() - t0) * 1000.0

    z_vals = inj_frame.pos[:, 2]
    print(
        f"Number of atoms to add: {num_atoms_to_add} (gaussian_spot "
        f"xy_3sigma={inject_xy_gaussian_3sigma}Å, sep={placement_distance_factor}×"
        f"{placement_bond_cutoff}Å, z_range=[{z_vals.min():.3f}, {z_vals.max():.3f}], "
        f"theta_sigma={theta_sigma_deg}°)"
    )
    if pipeline_timeit:
        print(
            f"gaussian_spot placement: {num_atoms_to_add} particles, "
            f"xy_sigma={xy_sigma:.2f}Å (3σ={inject_xy_gaussian_3sigma}Å), "
            f"sep={sep:.2f}Å, inject={timing.inject_ms:.2f} ms"
        )

    frame = frame.append(inj_frame)
    frame = crop_reserved_region(frame, box_x, box_y, box_z)

    t0 = time.perf_counter()
    write_model_xyz(
        frame,
        output_file,
        header,
        box_z,
        fast_io=fast_io,
        xyz_float_decimals=xyz_float_decimals,
    )
    timing.write_ms = (time.perf_counter() - t0) * 1000.0

    if pipeline_timeit:
        io_mode = "cpp" if (fast_io and _HAS_FAST_XYZ) else "python"
        print(
            f"[pipeline] io={io_mode} parse={timing.parse_ms:.0f}ms bfs={timing.bfs_ms:.0f}ms "
            f"surface={timing.surface_ms:.0f}ms inject={timing.inject_ms:.0f}ms "
            f"write={timing.write_ms:.0f}ms total={timing.total_ms:.0f}ms"
        )

    return timing, n_removed, frame.n
