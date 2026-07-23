#!/usr/bin/env python3
"""生成 v007 三维参数扫参配置：ts × v × gaussSpot3σ。"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(ROOT, "configs", "param_sweep_v007_theta5_antivel")
RUN_ROOT = "runs_param_sweep_v007_theta5_antivel"

TIMESTEPS = [1.0, 1.5, 2.0]
VELOCITIES = [0.005, 0.007, 0.0085, 0.01]
SIGMAS = [15, 20, 25, 30]


def tag_ts(ts):
    return str(ts).replace(".", "p")


def tag_v(v):
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


TEMPLATE = """# auto: ts={ts} v={v} sigma={sigma} bfs_cutoff=2.9 antivel_sep
run_subdir: "{run_subdir}"

Dirsstopsteps: 500
run_steps: 100
time_step: {ts}
box_z: 200
enable_d3: false
gpumd_command: "gpumd-4.7"
gpu_device: 0

substrate_temperature: 550
initial_xyz: "100.xyz"
substrate_replicate: 6

inject_species_weights:
  Ge: 1.0
  Si: 0.0

inject_mode: fast_ballistic_near_surface
fixed_xy_count: 1
fixed_xy_placement: center
fixed_xy_seed: 42
local_surface_radius: 5.0
z_spread: 0.0
cluster_cutoff: 3.2
remove_incident_particles: true

bfs_relaxed: true
bfs_cluster_cutoff: 2.9
bfs_near_surface_keep_angstrom: 35.0
bfs_light_percentile: 98.0
bfs_full_interval: 20

placement_bond_cutoff: 4.0
placement_distance_factor: 0.65
placement_offset_mode: antivel_sep
inject_xy_gaussian_3sigma: {sigma}.0
pipeline_timeit: true
surface_grid_cell_size: 5.0
inject_rng_base: 1000003
fast_io: true
xyz_float_decimals: 3

parser_mode: openmp
inject_fast: true

bfs_mode: light

velocity_magnitude: {v}
theta_sigma_deg: 5.0
inject_temperature: 2000
"""


def main():
    os.makedirs(CFG_DIR, exist_ok=True)
    manifest = os.path.join(CFG_DIR, "manifest.tsv")
    lines = ["id\tts\tv\tsigma\trun_subdir\tconfig\n"]
    idx = 0
    for ts in TIMESTEPS:
        for v in VELOCITIES:
            for sigma in SIGMAS:
                idx += 1
                run_id = f"ts{tag_ts(ts)}_v{tag_v(v)}_sig{sigma}"
                run_subdir = f"{RUN_ROOT}/{run_id}"
                cfg_name = f"param_{run_id}.yaml"
                cfg_path = os.path.join(CFG_DIR, cfg_name)
                with open(cfg_path, "w", encoding="utf-8") as f:
                    f.write(
                        TEMPLATE.format(
                            ts=ts, v=v, sigma=sigma, run_subdir=run_subdir
                        )
                    )
                lines.append(
                    f"{idx}\t{ts}\t{v}\t{sigma}\t{run_subdir}\t{cfg_path}\n"
                )
    with open(manifest, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Generated {idx} configs -> {CFG_DIR}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
