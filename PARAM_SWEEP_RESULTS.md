# v7 参数扫参结果（48 组）

单粒子中心注入，BFS `bfs_cluster_cutoff=2.9`，`placement_offset_mode=antivel_sep`。

## 扫参网格

| 维度 | 取值 |
|------|------|
| `time_step` | 1.0, 1.5, 2.0 fs |
| `velocity_magnitude` | 0.005, 0.007, 0.0085, 0.01 Å/fs |
| `inject_xy_gaussian_3sigma` | 15, 20, 25, 30 Å |

固定：`run_steps=100`, `theta=5°`, `Dirsstopsteps=500`, `rep6`, Ge100。

## 评价指标

1. **留存率** = final_Ge / 499（第 1 轮不加 Ge）
2. **岛心集中度** = 注入点正中央半径 20Å 内 Ge 占比
3. **综合分** = 留存 × (0.5 + 0.5 × 中心占比)

## Top 1 推荐参数（v8 生产沿用）

| 参数 | 值 |
|------|-----|
| `time_step` | **2.0 fs** |
| `run_steps` | 100 |
| `velocity_magnitude` | **0.007** Å/fs |
| `theta_sigma_deg` | 5° |
| `inject_xy_gaussian_3sigma` | **25 Å** |
| `placement_offset_mode` | antivel_sep |
| `bfs_cluster_cutoff` | 2.9 |
| `bfs_relaxed` | true (p98, near_surface_keep=35Å) |

**结果**：Ge=498，留存 99.8%，岛心 20Å 内 75.1%，BFS 删除 0。

## 三维趋势

| 维度 | 结论 |
|------|------|
| time_step | 越大越好：1.0→96.5%，2.0→99.3% |
| velocity | 0.007~0.01 最优；0.005 仅 95.1% |
| 3σ | 20~25Å 留存与中心兼顾；15Å 最差 |

## 原始数据

- 排名：[`runs_param_sweep_v007_theta5_antivel/analysis_ranking.tsv`](runs_param_sweep_v007_theta5_antivel/analysis_ranking.tsv)
- 计时：[`runs_param_sweep_v007_theta5_antivel/logs/sweep_timing.tsv`](runs_param_sweep_v007_theta5_antivel/logs/sweep_timing.tsv)
- 最优 run 目录：`runs_param_sweep_v007_theta5_antivel/ts2p0_v0p007_sig25/`

## 备选（留存相同）

- ts=2.0, v=0.0085, σ=20 → center20=74.1%
- ts=2.0, v=0.0085, σ=25 → center20=73.9%
