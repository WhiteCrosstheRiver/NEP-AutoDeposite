"""Numba-accelerated batch injection placement (surface zmax + intra-round overlap)."""

import math

import numpy as np

try:
    from numba import njit

    @njit(cache=True)
    def _pbc_mod_xy(x, box_len):
        r = x % box_len
        if r < 0.0:
            r += box_len
        return r

    @njit(cache=True)
    def _query_zmax_batch_numba(
        pos,
        order,
        starts,
        nx,
        ny,
        cell_size,
        box_x,
        box_y,
        global_zmax_pos,
        px_arr,
        py_arr,
        radius,
    ):
        n_q = px_arr.shape[0]
        radius_sq = radius * radius
        half_x = box_x * 0.5
        half_y = box_y * 0.5
        n_ring = max(1, int(math.ceil(radius / cell_size)))

        out = np.empty((n_q, 3), dtype=np.float64)
        g0 = global_zmax_pos[0]
        g1 = global_zmax_pos[1]
        g2 = global_zmax_pos[2]

        for qi in range(n_q):
            px = px_arr[qi]
            py = py_arr[qi]
            cx0 = int(math.floor(px / cell_size)) % nx
            cy0 = int(math.floor(py / cell_size)) % ny

            best_z = -1.0e300
            best_x = g0
            best_y = g1
            best_z_val = g2

            for dcx in range(-n_ring, n_ring + 1):
                for dcy in range(-n_ring, n_ring + 1):
                    cx = (cx0 + dcx) % nx
                    cy = (cy0 + dcy) % ny
                    nc = cx + nx * cy
                    start = starts[nc]
                    end = starts[nc + 1]
                    for k in range(start, end):
                        atom = order[k]
                        ax = pos[atom, 0]
                        ay = pos[atom, 1]
                        az = pos[atom, 2]
                        dx = abs(ax - px)
                        if dx > half_x:
                            dx = box_x - dx
                        dy = abs(ay - py)
                        if dy > half_y:
                            dy = box_y - dy
                        dist_sq = dx * dx + dy * dy
                        if dist_sq < radius_sq:
                            if az > best_z:
                                best_z = az
                                best_x = ax
                                best_y = ay
                                best_z_val = az

            out[qi, 0] = best_x
            out[qi, 1] = best_y
            out[qi, 2] = best_z_val
        return out

    @njit(cache=True)
    def _resolve_overlap_batch_numba(init_pos, vel, box_x, box_y, min_sep):
        n = init_pos.shape[0]
        out = np.empty((n, 3), dtype=np.float64)
        min_sep_sq = min_sep * min_sep
        golden = 2.399963229728653

        for i in range(n):
            pos0 = init_pos[i, 0]
            pos1 = init_pos[i, 1]
            pos2 = init_pos[i, 2]
            vx = vel[i, 0]
            vy = vel[i, 1]
            vz = vel[i, 2]

            v_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
            if v_norm < 1.0e-15 or i == 0:
                out[i, 0] = pos0
                out[i, 1] = pos1
                out[i, 2] = pos2
                continue

            vhx = vx / v_norm
            vhy = vy / v_norm
            vhz = vz / v_norm

            ux = vhy * 1.0 - vhz * 0.0
            uy = vhz * 0.0 - vhx * 1.0
            uz = vhx * 0.0 - vhy * 0.0
            u_norm = math.sqrt(ux * ux + uy * uy + uz * uz)
            if u_norm < 1.0e-8:
                ux = vhy * 0.0 - vhz * 0.0
                uy = vhz * 0.0 - vhx * 1.0
                uz = vhx * 1.0 - vhy * 0.0
                u_norm = math.sqrt(ux * ux + uy * uy + uz * uz)
            ux /= u_norm
            uy /= u_norm
            uz /= u_norm

            wx = vhy * uz - vhz * uy
            wy = vhz * ux - vhx * uz
            wz = vhx * uy - vhy * ux

            px = pos0
            py = pos1
            pz = pos2

            for step in range(24):
                too_close = False
                for j in range(i):
                    ox = out[j, 0]
                    oy = out[j, 1]
                    oz = out[j, 2]
                    dx = abs(px - ox)
                    if dx > box_x * 0.5:
                        dx = box_x - dx
                    dy = abs(py - oy)
                    if dy > box_y * 0.5:
                        dy = box_y - dy
                    dz = pz - oz
                    if dx * dx + dy * dy + dz * dz < min_sep_sq:
                        too_close = True
                        break
                if not too_close:
                    break
                phi = (step + 1) * golden
                px += 0.55 * (math.cos(phi) * ux + math.sin(phi) * wx)
                py += 0.55 * (math.cos(phi) * uy + math.sin(phi) * wy)
                pz += 0.55 * (math.cos(phi) * uz + math.sin(phi) * wz)

            out[i, 0] = px
            out[i, 1] = py
            out[i, 2] = pz
        return out

    _HAS_NUMBA_INJECT = True
except ImportError:
    _HAS_NUMBA_INJECT = False


def query_zmax_batch(surface_grid, px_arr, py_arr, radius):
    """Batch local zmax atom query; returns (Q, 3) coordinates."""
    if not _HAS_NUMBA_INJECT:
        raise RuntimeError("numba required for query_zmax_batch")
    px_arr = np.ascontiguousarray(px_arr, dtype=np.float64)
    py_arr = np.ascontiguousarray(py_arr, dtype=np.float64)
    pos = surface_grid.pos
    global_zmax_pos = pos[surface_grid.global_zmax_idx]
    return _query_zmax_batch_numba(
        pos,
        surface_grid._order,
        surface_grid._starts,
        surface_grid.nx,
        surface_grid.ny,
        surface_grid.cell_size,
        surface_grid.box_x,
        surface_grid.box_y,
        global_zmax_pos,
        px_arr,
        py_arr,
        float(radius),
    )


def resolve_overlap_batch(init_pos, vel, box_x, box_y, min_sep):
    """Batch intra-round overlap resolve; returns (N, 3) positions."""
    if not _HAS_NUMBA_INJECT:
        raise RuntimeError("numba required for resolve_overlap_batch")
    init_pos = np.ascontiguousarray(init_pos, dtype=np.float64)
    vel = np.ascontiguousarray(vel, dtype=np.float64)
    return _resolve_overlap_batch_numba(
        init_pos, vel, float(box_x), float(box_y), float(min_sep)
    )
