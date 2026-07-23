# v7_gaussSpotCenterSingleTest — 系统性工作流文档

## 一、项目概览

这是一个 **MBE (Molecular Beam Epitaxy) 分子束外延沉积模拟流水线**，使用 NEP (Neuroevolution Potential) 力场 + GPUMD 进行 SiGe 衬底上 Ge 岛生长的 MD 模拟。

### 版本演进路径

```
v1_cavity_simulation        → 腔体均匀喷淋
v2_test                     → 中心单点测试
v3_test                     → θ=1° 窄束流测试
v4_fixedRandomTenPointsLocalZ → 固定随机 XY、局部 Z 放置
v5_fastBallisticNearSurfaceDeposition → 弹道近表面注入（高斯斑）
v6_acceleratedNearSurfaceDeposition → SoA 数组化加速 + C++ IO
v7_gaussSpotCenterSingleTest → ★  当前：放宽 BFS + 反速度方向放置 + 参数扫参
v8_fixedRandomHundredAntivelDeposition → 生产：百点随机反速度沉积
```

---

## 二、核心文件架构

```
v7_gaussSpotCenterSingleTest/
├── demo.py                          # ★ 主驱动脚本（迭代循环）
├── pipeline_v7.py                   # ★ v7 专用流水线（放宽 BFS + antivel_sep）
├── inject_v7.py                     # ★ v7 注入变体（反速度方向放置 + Maxwell 速度）
├── center_xy.py                     # 衬底中心单点注入坐标
├── generate_param_sweep_configs.py  # 三维参数扫参配置生成
├── analyze_param_sweep.py           # 扫参结果分析与排名
│
├── run_single.sh                    # 单次运行入口
├── run_all_sweep*.sh               # 批量扫参入口（多种变体）
├── run_param_sweep_nohup.sh        # 参数扫参顺序执行
├── launch_*.sh                      # 各种 nohup 后台启动脚本
│
├── configs/                         # YAML 配置目录
│   └── param_sweep_v007_theta5_antivel/
│       ├── manifest.tsv             # 扫参清单
│       └── param_*.yaml             # 各组参数配置
│
├── runs_param_sweep_v007_theta5_antivel/  # 扫参结果输出
│   └── analysis_ranking.tsv         # ★ 排名结果
│
├── 100.xyz                          # 初始衬底结构
├── nep.txt                          # NEP 力场文件
└── README.md / PARAM_SWEEP_RESULTS.md

v6_acceleratedNearSurfaceDeposition/ (核心引擎库)
├── array_pipeline.py                # ★ SoA 数组流水线（解析/过滤/注入/写入）
├── utility.py                       # ★ 物理工具（BFS/NEP/速度采样/坐标管理）
├── surface_grid.py                  # 2D 表面网格（快速空间查询）
├── inject_fast.py                   # numba 批量注入加速
├── fast_xyz.cpython-312-*.so        # C++ 快速 IO 扩展
├── build_fast_xyz.sh               # C++ 扩展编译脚本
└── setup.py
```

---

## 三、完整工作流（Step by Step）

### 阶段 0：环境准备

```bash
# 0.1 编译 C++ 加速模块（首次使用）
cd v6_acceleratedNearSurfaceDeposition
bash build_fast_xyz.sh          # 生成 fast_xyz.cpython-312-*.so

# 0.2 确认 GPUMD 可用
which gpumd-4.7                 # GPU MD 引擎
nvidia-smi                       # 确认 GPU 可用

# 0.3 确认初始结构
ls 100.xyz                       # 初始 SiGe 衬底模型（extended xyz 格式）
ls nep.txt                       # NEP 力场参数文件
```

### 阶段 1：单次运行 (debug / 验证)

**Step 1.1 — 编写 YAML 配置**

```yaml
# configs/my_test.yaml
run_subdir: "my_test_run"

Dirsstopsteps: 500          # 总沉积轮数 (每轮加1个Ge，共499个)
run_steps: 100              # GPUMD 每轮运行步数
time_step: 2.0              # MD 时间步长 (fs)
box_z: 200                  # Z 方向盒子尺寸 (Å)
enable_d3: false            # 是否启用 DFT-D3 修正
gpumd_command: "gpumd-4.7"  # GPUMD 可执行文件
gpu_device: 0               # GPU 设备编号

substrate_temperature: 550  # 衬底温度 (K)
initial_xyz: "100.xyz"      # 初始结构文件
substrate_replicate: 6      # XY 方向复制倍数 (6×6=36倍面积)

inject_species_weights:     # 注入物种
  Ge: 1.0
  Si: 0.0

inject_mode: fast_ballistic_near_surface  # 注入模式
fixed_xy_count: 1           # 每轮固定注入点数
fixed_xy_placement: center  # 注入点位置：center=衬底正中央
fixed_xy_seed: 42           # 固定点随机种子
local_surface_radius: 5.0   # 局部表面搜索半径 (Å)
cluster_cutoff: 3.2         # 主沉积体聚类截断 (Å)
remove_incident_particles: true  # 是否剔除孤立入射粒子

bfs_relaxed: true           # 启用放宽 BFS (v7 核心特性)
bfs_cluster_cutoff: 2.9     # BFS 聚类截断 (Å)
bfs_near_surface_keep_angstrom: 35.0  # 近表面保留范围 (Å)
bfs_light_percentile: 98.0  # light BFS 分位阈值
bfs_full_interval: 20       # 每 N 轮全图 BFS 校准一次

placement_bond_cutoff: 4.0           # 放置键截断 (Å)
placement_distance_factor: 0.65      # 放置距离因子
placement_offset_mode: antivel_sep   # ★ 反速度方向偏移放置
inject_xy_gaussian_3sigma: 25.0      # 注入 XY 高斯斑 3σ 宽度 (Å)

velocity_magnitude: 0.007  # 注入速度模长 (Å/fs)
theta_sigma_deg: 5.0       # 束流发散角 σ_θ (度)
inject_velocity_mode: fixed_magnitude  # 速度模式
inject_temperature: 2000   # Maxwell 模式下的温度 (K)

pipeline_timeit: true      # 输出流水线计时
surface_grid_cell_size: 5.0  # 表面网格单元尺寸
inject_rng_base: 1000003   # 注入随机数基数
fast_io: true              # 启用 C++ 快速 IO
xyz_float_decimals: 3      # XYZ 输出小数位
parser_mode: openmp        # 解析器模式
inject_fast: true          # 启用 numba 批量注入加速
bfs_mode: light            # BFS 模式: light/shadow/full
```

**Step 1.2 — 运行**

```bash
cd v7_gaussSpotCenterSingleTest

# 方式 A：直接运行
python3 demo.py configs/my_test.yaml

# 方式 B：通过 wrapper 脚本运行
bash run_single.sh 0 25    # GPU=0, sigma=25

# 方式 C：后台 nohup 运行
nohup python3 demo.py configs/my_test.yaml > run.log 2>&1 &
```

**Step 1.3 — 运行时监控**

```bash
# 查看进度（当前轮数）
ls my_test_run/ | sort -n | tail -5

# 查看实时日志
tail -f run.log

# 检查 GPU 利用率
nvidia-smi
```

### 阶段 2：参数扫参 (grid search)

**Step 2.1 — 生成配置**

```bash
cd v7_gaussSpotCenterSingleTest
python3 generate_param_sweep_configs.py
```

这会生成 3 维网格配置：
- `time_step` ∈ {1.0, 1.5, 2.0} fs
- `velocity_magnitude` ∈ {0.005, 0.007, 0.0085, 0.01} Å/fs
- `inject_xy_gaussian_3sigma` ∈ {15, 20, 25, 30} Å

共 3×4×4 = 48 组，输出到 `configs/param_sweep_v007_theta5_antivel/`

**Step 2.2 — 启动扫参**

```bash
# 后台启动（单 GPU 顺序执行 48 组）
bash launch_param_sweep_nohup.sh 0

# 监控进度
tail -f runs_param_sweep_v007_theta5_antivel/logs/nohup_sweep_all.log
```

**Step 2.3 — 随时查看排名**

```bash
python3 analyze_param_sweep.py
```

输出排名到 `runs_param_sweep_v007_theta5_antivel/analysis_ranking.tsv`

### 阶段 3：结果分析

**Step 3.1 — 分析脚本输出指标**

`analyze_param_sweep.py` 对每组参数的最后 dump.xyz 计算：

| 指标 | 含义 | 目标 |
|------|------|------|
| `retention` | 最终 Ge 数 / 注入总数 (499) | 越大越好 (→1.0) |
| `frac_center_20A` | 岛心 20Å 半径内 Ge 占比 | 越大越好 (→1.0) |
| `mean_r_xy` | Ge 原子距岛心平均 XY 距离 | 越小越好 |
| `score` | 综合分 = retention×(0.5+0.5×frac_center) | 越大越好 |

**Step 3.2 — 最优参数选取**

从 `PARAM_SWEEP_RESULTS.md` 结论：

| 参数 | 最优值 |
|------|--------|
| `time_step` | **2.0 fs** |
| `velocity_magnitude` | **0.007 Å/fs** |
| `inject_xy_gaussian_3sigma` | **25 Å** |
| `bfs_cluster_cutoff` | **2.9** |
| `placement_offset_mode` | **antivel_sep** |

结果：Ge=498, 留存=99.8%, 岛心20Å内=75.1%, BFS删除=0

---

## 四、数据流详解（单轮迭代）

```
Round N dump.xyz (extended xyz, 含 Si+Ge 原子)
        │
        ▼
┌─ parse_dump_soa() ─────────────────────────────┐
│  fast_xyz.read_dump_xyz() [C++] 或 Python 解析  │
│  → AtomFrame(species, pos, vel, group)         │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─ filter_main_deposit_soa_relaxed() ─────────────┐
│  1. resolve_bfs_keep_mask()                     │
│     - light mode: z>percentile 的 candidate     │
│       仅保留与低z基体连通者                      │
│     - 每 bfs_full_interval 轮全图 BFS 校准       │
│  2. 放宽：z ≤ substrate_top+35Å 的被删原子救回  │
│  → 过滤后的 AtomFrame                           │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─ apply_group_labels() ──────────────────────────┐
│  衬底底部 (z∈[0,3]) 原子标记 group=1 (固定)     │
│  其余 group=0 (自由)                             │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─ SurfaceGrid2D(dep_pos) ────────────────────────┐
│  在 XY 平面上构建网格索引，用于快速局部查询       │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─ generate_injections_soa_antivel_sep() ─────────┐
│  v7 核心：沿速度反方向放置                       │
│  1. 每固定点 XY 高斯洒点 (σ = 3σ/3)             │
│  2. 束流速度：|v|固定 或 Maxwell(T) 采样         │
│     θ ~ |N(0, σ_θ)|, φ ~ U(0, 2π), 主束 -Z     │
│  3. 锚点：洒点 XY + 局部 zmax (5Å 搜索半径)      │
│  4. 放置：pos = anchor - separation * v_hat     │
│  5. 同轮去重叠：垂直于 v 的切向微调              │
│  → AtomFrame(inj_species, inj_pos, inj_vel)    │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─ frame.append(inj_frame) ───────────────────────┐
│  将注入原子拼接到现有体系                         │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─ crop_reserved_region() ────────────────────────┐
│  裁剪超出 RESERVED_REGION 的原子                 │
│  (z > box_z-50 的删除)                          │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─ write_model_xyz() ─────────────────────────────┐
│  fast_xyz.write_model_xyz() [C++] 或 Python     │
│  → Round N+1 model.xyz                          │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─ generate_run_in() ─────────────────────────────┐
│  生成 GPUMD 输入文件 run.in                       │
│  - potential ../nep.txt                         │
│  - time_step, ensemble nvt_ber, run_steps       │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─ GPUMD (subprocess) ────────────────────────────┐
│  CUDA_VISIBLE_DEVICES=X gpumd-4.7               │
│  输入: model.xyz + run.in + nep.txt              │
│  输出: dump.xyz (下一轮输入)                     │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
              Round N+1 dump.xyz
```

---

## 五、v7 相比 v6 的关键改进

| 特性 | v6 | v7 |
|------|----|----|
| BFS 过滤 | 标准 BFS，可能误删刚沉积的 Ge | **放宽 BFS**：z ≤ Si_top + margin 的原子救回 |
| 放置方向 | +Z 方向偏移 | **反速度方向偏移** (antivel_sep)：沿 -v_hat 放置 |
| 发射源 | 无 | 可选 fixed_z 发射源锚定 (antivel_sep_fixed_z) |
| 速度采样 | |v| 固定 | 新增 **Maxwell 热分布** (maxwell_theta)：|v| ~ χ(3) × sqrt(kT/m) |
| 扫参 | 手动 | **自动三维扫参** + 自动排名分析 |

---

## 六、配置参数详细说明

### 注入模式

| inject_mode | 说明 |
|-------------|------|
| `fast_ballistic_near_surface` | ★ v5-v8 标准：高斯斑 XY + 近表面放置 |
| `fixed_random_xy_local_z` | v4 模式：固定随机 XY，局部 Z 偏移 |
| `center_single` | v2 模式：衬底正中央单点 |
| 其他 (cavity) | v1 模式：均匀网格喷淋 |

### 放置偏移模式

| placement_offset_mode | 说明 |
|-----------------------|------|
| `z_plus` (默认) | pos = anchor + (0, 0, sep)，垂直向上 |
| `antivel_sep` ★ | pos = anchor - sep × v_hat，沿速度反方向 |
| `antivel_sep_fixed_z` | 同上 + 发射源固定在 z=inject_z_anchor |

### BFS 模式

| bfs_mode | 说明 |
|----------|------|
| `full` | 每次全图 BFS（慢，准确） |
| `light` ★ | candidate-shell 快速模式，每 N 轮全校准 |
| `shadow` | 同时算两种，输出用 full，记录差异用于验证 |

### 速度模式

| inject_velocity_mode | 说明 |
|----------------------|------|
| `fixed_magnitude` (默认) | |v| = velocity_magnitude 固定 |
| `maxwell_theta` | |v| ~ Maxwell(T)，方向仍为高斯束流 |

---

## 七、常见操作速查

```bash
# === 环境 ===
cd /home/gpu02/fzm/paper/1.MBE-SiGe/4.confiure_islands_useNEP/1.UseNEPandUse/v7_gaussSpotCenterSingleTest

# === 生成扫参配置 ===
python3 generate_param_sweep_configs.py

# === 启动扫参 ===
bash launch_param_sweep_nohup.sh 0                          # GPU 0
tail -f runs_param_sweep_v007_theta5_antivel/logs/nohup_sweep_all.log

# === 检查排名 ===
python3 analyze_param_sweep.py

# === 单次运行 ===
python3 demo.py configs/param_sweep_v007_theta5_antivel/param_ts2p0_v0p007_sig25.yaml

# === 批量运行特定脚本 ===
bash run_all_sweep_relaxed_run50_v005_theta60_antivel_nohup.sh 0  # GPU 0

# === 查看结果 ===
cat runs_param_sweep_v007_theta5_antivel/analysis_ranking.tsv
cat PARAM_SWEEP_RESULTS.md

# === 从最优 run 目录查看最终 dump ===
head -5 runs_param_sweep_v007_theta5_antivel/ts2p0_v0p007_sig25/500/dump.xyz
```

---

## 八、产出物 → 下一步 (v8)

v7 扫参确定最优参数后，v8 (`v8_fixedRandomHundredAntivelDeposition`) 将单点中心注入扩展为 **100 个随机固定点**，使用相同的 antivel_sep 放置策略进行生产级沉积模拟。

v8 的关键变化：
- `fixed_xy_count: 100`（每轮 100 个 Ge）
- `fixed_xy_placement: random`（而非 center）
- 延续 v7 的最优参数：ts=2.0, v=0.007, sigma=25, bfs_cutoff=2.9
