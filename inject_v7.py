"""v7 注入变体：沿速度反方向放置（不修改 v6 inject）。"""

import math
import random

import numpy as np

from array_pipeline import AtomFrame, SPECIES_TO_INT, generate_injections_soa
from utility import (
    AMU,
    KB,
    M_S_TO_ANGSTROM_FS,
    _pbc_wrap_xy,
    generate_gaussian_beam_velocity,
)

try:
    from inject_fast import query_zmax_batch, resolve_overlap_batch
except ImportError:
    query_zmax_batch = None
    resolve_overlap_batch = None

_ATOMIC_MASSES_AMU = {
    "Ge": 72.630,
    "Si": 28.085,
    "Ga": 69.723,
    "N": 14.007,
}


def sample_maxwell_speed_magnitude(element_symbol, temperature_k, rng):
    """3D Maxwell 速率分布采样 |v|，返回 Å/fs。"""
    if element_symbol not in _ATOMIC_MASSES_AMU:
        raise ValueError(f"未知元素符号: {element_symbol}")
    mass_kg = _ATOMIC_MASSES_AMU[element_symbol] * AMU
    sigma_ms = math.sqrt(KB * float(temperature_k) / mass_kg)
    v_ms = sigma_ms * math.sqrt(float(rng.chisquare(3)))
    return v_ms * M_S_TO_ANGSTROM_FS


def generate_gaussian_beam_velocity_maxwell(
    temperature_k,
    element_symbol="Ge",
    theta_sigma_deg=5.0,
    rng=None,
):
    """
    |v| ~ Maxwell(T)，方向 θ~|N(0,σ_θ)|、φ~U(0,2π)，主束 -Z。
    返回 (vx, vy, vz, v_mag) 单位 Å/fs。
    """
    if rng is None:
        rng = np.random.default_rng()
    v_mag = sample_maxwell_speed_magnitude(element_symbol, temperature_k, rng)
    theta_sigma_rad = math.radians(float(theta_sigma_deg))
    theta = abs(float(rng.normal(0.0, theta_sigma_rad)))
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    vx = v_mag * math.sin(theta) * math.cos(phi)
    vy = v_mag * math.sin(theta) * math.sin(phi)
    vz = -v_mag * math.cos(theta)
    return vx, vy, vz, v_mag

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
    inject_z_anchor=None,
    inject_velocity_mode="fixed_magnitude",
    inject_temperature=None,
):
    """
    高斯斑 XY + 高斯束流速度；表面锚点 + 沿 -v_hat 偏移 sep 放置。

    inject_z_anchor 为 None：近表面模式（锚点 Z = 局部 zmax）。
    inject_z_anchor 给定（如 200Å）：发射源固定在 z=inject_z_anchor；
    落点仍按 (px,py) 高斯洒点 → 5Å 内 zmax → anchor → pos=anchor-sep*v_hat。
    同时由 anchor 沿 -v_hat 反推发射源坐标，用于飞行路径计算/日志。
    """
    if inject_species_weights is None:
        inject_species_weights = {"Ge": 1.0, "Si": 0.0}

    sep = float(placement_distance_factor) * float(placement_bond_cutoff)
    xy_sigma = float(inject_xy_gaussian_3sigma) / 3.0
    min_intra_sep = max(2.8, sep * 0.85)
    v_mag_fixed = float(velocity_magnitude) if velocity_magnitude is not None else 0.007
    vel_mode = str(inject_velocity_mode).lower()
    n_inj = len(fixed_xy_points)

    species_list = list(inject_species_weights.keys())
    weights_list = list(inject_species_weights.values())

    inj_species = np.empty(n_inj, dtype=np.int8)
    inj_vel = np.empty((n_inj, 3), dtype=np.float64)
    px_arr = np.empty(n_inj, dtype=np.float64)
    py_arr = np.empty(n_inj, dtype=np.float64)

    v_mag_samples = []

    for i, (px0, py0) in enumerate(fixed_xy_points):
        sp_name = random.choices(species_list, weights=weights_list)[0]
        inj_species[i] = SPECIES_TO_INT.get(sp_name, 1)

        if vel_mode == "maxwell_theta":
            if inject_temperature is None:
                raise ValueError("maxwell_theta 模式需要 inject_temperature (K)")
            vx, vy, vz, v_mag_i = generate_gaussian_beam_velocity_maxwell(
                inject_temperature,
                element_symbol=sp_name,
                theta_sigma_deg=theta_sigma_deg,
                rng=rng,
            )
            v_mag_samples.append(v_mag_i)
        else:
            vx, vy, vz = generate_gaussian_beam_velocity(
                v_mag_fixed,
                theta_sigma_deg=theta_sigma_deg,
                rng=rng,
            )
            v_mag_samples.append(v_mag_fixed)

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

    emit_info = None

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

        if inject_z_anchor is not None:
            z_emit = float(inject_z_anchor)
            vz_hat = v_hat[:, 2]
            dz = anchor[:, 2] - z_emit
            # 束流主方向向下；若个别粒子 vz>=0 则退回仅用洒点 XY 的发射高度
            t_flight = np.empty(n_inj, dtype=np.float64)
            emission = np.empty((n_inj, 3), dtype=np.float64)
            for i in range(n_inj):
                if vz_hat[i] < -1e-12:
                    t_flight[i] = dz[i] / vz_hat[i]
                    emission[i] = anchor[i] - t_flight[i] * v_hat[i]
                else:
                    t_flight[i] = 0.0
                    emission[i, 0] = px_arr[i]
                    emission[i, 1] = py_arr[i]
                    emission[i, 2] = z_emit
            emit_info = {
                "z_emit": z_emit,
                "emission": emission,
                "t_flight": t_flight,
                "anchor": anchor.copy(),
            }

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
        return frame, sep_fb, xy_sigma_fb, None, None

    inj_group = np.zeros(n_inj, dtype=np.uint8)
    v_mag_info = {
        "mode": vel_mode,
        "samples": v_mag_samples,
        "temperature_k": inject_temperature,
    }
    return (
        AtomFrame(species=inj_species, pos=inj_pos, vel=inj_vel, group=inj_group),
        sep,
        xy_sigma,
        emit_info,
        v_mag_info,
    )
