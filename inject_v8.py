"""v7 注入变体：沿速度反方向放置（不修改 v6 inject）。"""

import random

import numpy as np

from array_pipeline import AtomFrame, SPECIES_TO_INT, generate_injections_soa
from utility import _pbc_wrap_xy, generate_gaussian_beam_velocity

try:
    from inject_fast import query_zmax_batch, resolve_overlap_batch
except ImportError:
    query_zmax_batch = None
    resolve_overlap_batch = None


def generate_injections_soa_antivel_sep(
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
    """
    近表面注入：锚点为落点 (px, py, z_local_max)。
    粒子位置 = anchor - sep * v_hat（沿速度反方向偏移 sep），|v| 不变。
    """
    if inject_species_weights is None:
        inject_species_weights = {"Ge": 1.0, "Si": 0.0}

    sep = float(placement_distance_factor) * float(placement_bond_cutoff)
    xy_sigma = float(inject_xy_gaussian_3sigma) / 3.0
    min_intra_sep = max(2.8, sep * 0.85)
    v_mag = float(velocity_magnitude)
    n_inj = len(fixed_xy_points)

    species_list = list(inject_species_weights.keys())
    weights_list = list(inject_species_weights.values())

    inj_species = np.empty(n_inj, dtype=np.int8)
    inj_vel = np.empty((n_inj, 3), dtype=np.float64)
    px_arr = np.empty(n_inj, dtype=np.float64)
    py_arr = np.empty(n_inj, dtype=np.float64)

    for i, (px0, py0) in enumerate(fixed_xy_points):
        vx, vy, vz = generate_gaussian_beam_velocity(
            v_mag,
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

    if inject_fast and query_zmax_batch is not None and resolve_overlap_batch is not None:
        zmax_atoms = query_zmax_batch(
            surface_grid, px_arr, py_arr, float(local_search_radius)
        )
        anchor = np.empty((n_inj, 3), dtype=np.float64)
        anchor[:, 0] = px_arr
        anchor[:, 1] = py_arr
        anchor[:, 2] = zmax_atoms[:, 2]

        v_norm = np.linalg.norm(inj_vel, axis=1, keepdims=True)
        v_norm = np.maximum(v_norm, 1e-30)
        v_hat = inj_vel / v_norm
        init_pos = anchor - sep * v_hat
        inj_pos = resolve_overlap_batch(
            init_pos, inj_vel, float(box_x), float(box_y), min_intra_sep
        )
    else:
        # 回退：仍用 v6 默认 +Z，但不应在 v7 正常环境触发
        frame, sep_fb, xy_sigma_fb = generate_injections_soa(
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
            inject_fast=False,
        )
        return frame, sep_fb, xy_sigma_fb

    inj_group = np.zeros(n_inj, dtype=np.uint8)
    return AtomFrame(
        species=inj_species, pos=inj_pos, vel=inj_vel, group=inj_group
    ), sep, xy_sigma
