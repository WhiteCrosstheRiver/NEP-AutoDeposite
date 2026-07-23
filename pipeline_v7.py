"""v7 专用：放宽 BFS 入射粒子清理（不修改 v6 array_pipeline）。"""

import time

import numpy as np

from array_pipeline import (
    PipelineTiming,
    apply_group_labels,
    crop_reserved_region,
    generate_injections_soa,
    parse_dump_soa,
    write_model_xyz,
)
from inject_v7 import generate_injections_soa_antivel_sep
from utility import format_bfs_light_log, load_fixed_inject_xy, resolve_bfs_keep_mask


def filter_main_deposit_soa_relaxed(
    frame,
    box_x,
    box_y,
    cluster_cutoff,
    bfs_mode="light",
    bfs_full_interval=10,
    bfs_light_percentile=90.0,
    round_num=1,
    near_surface_keep_angstrom=35.0,
):
    """
    在标准 BFS 之后，把 z 仍落在衬底顶面 + margin 内的被删原子救回。
    避免刚沉积、尚未在 cutoff 内成键的近表面 Ge 被误删。
    """
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

    margin = float(near_surface_keep_angstrom)
    if margin > 0.0:
        removed = ~keep_mask
        if removed.any():
            pos = frame.pos
            si_mask = frame.species == 0
            if si_mask.any():
                substrate_top = float(np.max(pos[si_mask, 2]))
            else:
                substrate_top = float(np.min(pos[:, 2]))
            z_keep_max = substrate_top + margin
            rescue = removed & (pos[:, 2] <= z_keep_max)
            n_rescue = int(rescue.sum())
            if n_rescue:
                keep_mask = keep_mask | rescue
                print(
                    f"[bfs-relaxed] kept {n_rescue} near-surface atom(s) "
                    f"(z<={z_keep_max:.2f}Å, Si_top={substrate_top:.2f}Å, margin={margin:.1f}Å)"
                )

    n_removed = int((~keep_mask).sum())
    if stats.mode != "shadow":
        stats.n_removed = n_removed
    else:
        stats.full_removed = n_removed
    return frame.slice_mask(keep_mask), n_removed, stats


def last_converted_new_soa_relaxed(
    file_path,
    output_file,
    box_z=200.0,
    inject_species_weights=None,
    local_surface_radius=5.0,
    cluster_cutoff=3.2,
    bfs_cluster_cutoff=None,
    remove_incident_particles=True,
    bfs_mode="light",
    bfs_full_interval=10,
    bfs_light_percentile=90.0,
    bfs_near_surface_keep_angstrom=35.0,
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
    placement_offset_mode="z_plus",
    inject_z_anchor=None,
    inject_velocity_mode="fixed_magnitude",
    inject_temperature=None,
    **_ignored,
):
    """v6 last_converted_new_soa + 放宽 BFS + 可选 antivel_sep 放置。"""
    import os

    from surface_grid import SurfaceGrid2D

    bfs_cut = float(bfs_cluster_cutoff if bfs_cluster_cutoff is not None else cluster_cutoff)
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
        frame, n_removed, bfs_stats = filter_main_deposit_soa_relaxed(
            frame,
            box_x,
            box_y,
            bfs_cut,
            bfs_mode=bfs_mode,
            bfs_full_interval=bfs_full_interval,
            bfs_light_percentile=bfs_light_percentile,
            round_num=round_num,
            near_surface_keep_angstrom=bfs_near_surface_keep_angstrom,
        )
        timing.bfs_ms = (time.perf_counter() - t0) * 1000.0
        if n_removed:
            print(
                f"Removed {n_removed} disconnected incident particle(s) "
                f"(bfs_cluster_cutoff={bfs_cut}, bfs_mode={bfs_mode}, "
                f"near_surface_keep={bfs_near_surface_keep_angstrom}Å)"
            )
        if bfs_stats is not None and bfs_mode in ("light", "shadow"):
            print(format_bfs_light_log(bfs_stats, round_num, frame.n + n_removed))

    frame = apply_group_labels(frame, box_x, box_y)
    dep_pos = frame.pos

    if inject_mode != "fast_ballistic_near_surface":
        raise ValueError(
            f"v7 relaxed pipeline 仅支持 fast_ballistic_near_surface，got {inject_mode!r}"
        )

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
    else:
        rng = np.random.default_rng()

    placement_mode = str(placement_offset_mode).lower()
    inject_fn = generate_injections_soa
    inject_fn_kwargs = {}
    if placement_mode == "antivel_sep":
        inject_fn = generate_injections_soa_antivel_sep
    elif placement_mode == "antivel_sep_fixed_z":
        inject_fn = generate_injections_soa_antivel_sep
        if inject_z_anchor is None:
            raise ValueError(
                "antivel_sep_fixed_z 需要在 config 中设置 inject_z_anchor（如 200.0）"
            )
        inject_fn_kwargs["inject_z_anchor"] = float(inject_z_anchor)

    t0 = time.perf_counter()
    inj_result = inject_fn(
        fixed_xy_points=fixed_xy_points,
        surface_grid=surface_grid,
        box_x=box_x,
        box_y=box_y,
        local_search_radius=float(local_surface_radius),
        velocity_magnitude=velocity_magnitude,
        theta_sigma_deg=float(theta_sigma_deg),
        placement_bond_cutoff=float(placement_bond_cutoff),
        placement_distance_factor=float(placement_distance_factor),
        inject_xy_gaussian_3sigma=float(inject_xy_gaussian_3sigma),
        inject_species_weights=inject_species_weights,
        rng=rng,
        inject_fast=inject_fast,
        inject_velocity_mode=inject_velocity_mode,
        inject_temperature=inject_temperature,
        **inject_fn_kwargs,
    )
    emit_info = None
    v_mag_info = None
    if placement_mode in ("antivel_sep", "antivel_sep_fixed_z"):
        inj_frame, sep, xy_sigma, emit_info, v_mag_info = inj_result
    else:
        inj_frame, sep, xy_sigma = inj_result
    timing.inject_ms = (time.perf_counter() - t0) * 1000.0

    z_vals = inj_frame.pos[:, 2]
    vel_norms = np.linalg.norm(inj_frame.vel, axis=1)
    if v_mag_info and str(v_mag_info.get("mode")) == "maxwell_theta":
        v_desc = (
            f"|v|~Maxwell(T={float(v_mag_info['temperature_k']):.0f}K), "
            f"sample=[{vel_norms.min():.4f},{vel_norms.max():.4f}]Å/fs"
        )
    else:
        v_desc = f"|v|={velocity_magnitude}"
    if placement_mode == "antivel_sep_fixed_z":
        offset_desc = (
            f"antivel_sep={sep:.2f}Å along -v_hat, "
            f"emit_src_z={float(inject_z_anchor):.1f}Å (surface anchor Z)"
        )
    elif placement_mode == "antivel_sep":
        offset_desc = f"antivel_sep={sep:.2f}Å along -v_hat"
    else:
        offset_desc = f"sep={placement_distance_factor}×{placement_bond_cutoff}Å (+Z)"
    print(
        f"Number of atoms to add: {num_atoms_to_add} (gaussian_spot "
        f"xy_3sigma={inject_xy_gaussian_3sigma}Å, {offset_desc}, "
        f"z_range=[{z_vals.min():.3f}, {z_vals.max():.3f}], "
        f"{v_desc}, theta_sigma={theta_sigma_deg}°)"
    )
    if pipeline_timeit:
        print(
            f"gaussian_spot placement: {num_atoms_to_add} particles, "
            f"mode={placement_mode}, xy_sigma={xy_sigma:.2f}Å "
            f"(3σ={inject_xy_gaussian_3sigma}Å), sep={sep:.2f}Å, "
            f"inject={timing.inject_ms:.2f} ms"
        )
    if emit_info is not None and num_atoms_to_add > 0:
        em = emit_info["emission"][0]
        anc = emit_info["anchor"][0]
        tf = float(emit_info["t_flight"][0])
        flight_len = float(np.linalg.norm(anc - em))
        print(
            f"[emit-src] z_emit={emit_info['z_emit']:.1f}Å "
            f"emit=({em[0]:.2f},{em[1]:.2f},{em[2]:.2f}) "
            f"surface_anchor=({anc[0]:.2f},{anc[1]:.2f},{anc[2]:.2f}) "
            f"flight_t={tf:.1f}Å_path, flight_len={flight_len:.1f}Å"
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
        from array_pipeline import _HAS_FAST_XYZ

        io_mode = "cpp" if (fast_io and _HAS_FAST_XYZ) else "python"
        print(
            f"[pipeline] io={io_mode} parse={timing.parse_ms:.0f}ms bfs={timing.bfs_ms:.0f}ms "
            f"surface={timing.surface_ms:.0f} inject={timing.inject_ms:.0f}ms "
            f"write={timing.write_ms:.0f}ms total={timing.total_ms:.0f}ms"
        )

    return timing, n_removed, frame.n
