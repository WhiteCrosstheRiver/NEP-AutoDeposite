"""2D cell list for local surface Zmax queries (PBC in XY)."""

import numpy as np


class SurfaceGrid2D:
    """Bucket atoms by XY cell; query local Zmax within a circular radius."""

    def __init__(self, pos, box_x, box_y, cell_size=5.0):
        self.box_x = float(box_x)
        self.box_y = float(box_y)
        self.cell_size = float(cell_size)
        self.nx = max(1, int(np.ceil(self.box_x / self.cell_size)))
        self.ny = max(1, int(np.ceil(self.box_y / self.cell_size)))
        self.pos = np.ascontiguousarray(pos, dtype=np.float64)
        n_atoms = self.pos.shape[0]
        self.global_zmax_idx = int(np.argmax(self.pos[:, 2])) if n_atoms > 0 else -1

        cell_ids = (
            (np.floor(self.pos[:, 0] / self.cell_size).astype(np.int64) % self.nx)
            + self.nx
            * (np.floor(self.pos[:, 1] / self.cell_size).astype(np.int64) % self.ny)
        )
        order = np.argsort(cell_ids, kind="mergesort")
        sorted_ids = cell_ids[order]
        n_cells = self.nx * self.ny
        starts = np.zeros(n_cells + 1, dtype=np.int64)
        np.add.accumulate(np.bincount(sorted_ids, minlength=n_cells), out=starts[1:])
        self._order = order
        self._starts = starts

    def _cell_range(self, cx, cy):
        nc = int(cx + self.nx * cy)
        return int(self._starts[nc]), int(self._starts[nc + 1])

    def query_zmax_atom(self, px, py, radius=5.0):
        """Return coordinates of highest-z atom within radius (PBC XY); else global max."""
        if self.pos.shape[0] == 0:
            return np.array([float(px), float(py), 0.0], dtype=np.float64)

        radius_sq = float(radius) ** 2
        half_x = self.box_x * 0.5
        half_y = self.box_y * 0.5
        px = float(px)
        py = float(py)

        cx0 = int(np.floor(px / self.cell_size)) % self.nx
        cy0 = int(np.floor(py / self.cell_size)) % self.ny
        n_ring = max(1, int(np.ceil(float(radius) / self.cell_size)))

        best_z = -np.inf
        best = self.pos[self.global_zmax_idx].copy()

        for dcx in range(-n_ring, n_ring + 1):
            for dcy in range(-n_ring, n_ring + 1):
                cx = (cx0 + dcx) % self.nx
                cy = (cy0 + dcy) % self.ny
                start, end = self._cell_range(cx, cy)
                if start == end:
                    continue
                idxs = self._order[start:end]
                pts = self.pos[idxs]
                dx = np.abs(pts[:, 0] - px)
                dy = np.abs(pts[:, 1] - py)
                dx = np.where(dx > half_x, self.box_x - dx, dx)
                dy = np.where(dy > half_y, self.box_y - dy, dy)
                dist_sq = dx * dx + dy * dy
                local = dist_sq < radius_sq
                if not np.any(local):
                    continue
                pool = pts[local]
                zmax_local = float(np.max(pool[:, 2]))
                if zmax_local > best_z:
                    best_z = zmax_local
                    best = pool[int(np.argmax(pool[:, 2]))].copy()

        if best_z == -np.inf:
            return self.pos[self.global_zmax_idx].copy()
        return best
