"""
批量任务准备：读取 total_config.yaml，笛卡尔积展开参数，
为每个任务目录写入 config.yaml、拷贝所需文件。
"""
import itertools
import os
import shutil
import yaml


def _ensure_list(v):
    """若为单值则转为单元素列表，便于统一做笛卡尔积。"""
    if v is None:
        return [None]
    if isinstance(v, list):
        return v
    return [v]


def _species_weights_to_dict(item):
    """从 total_config 的 inject_species_weights 的一项得到 demo 可用的 dict（去掉 name）。"""
    if isinstance(item, dict):
        return {k: v for k, v in item.items() if k != "name"}
    return item


def resolve_initial_xyz_dir(source_root, total_config):
    """
    解析衬底 xyz 所在目录。
    - total_config.initial_xyz_dir：相对 source_root 或绝对路径
    - 默认 Substrate
    """
    rel = total_config.get("initial_xyz_dir", "Substrate")
    if not rel:
        rel = "Substrate"
    if os.path.isabs(rel):
        return rel
    return os.path.join(source_root, rel)


def load_total_config(config_path):
    """加载 total_config.yaml，返回解析后的 dict。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_tasks(total_config):
    """
    根据 total_config 的变参数做笛卡尔积，生成 (index, task_config, dir_name) 列表。
    task_config 为单任务用的 config 字典（可直接写 config.yaml）；
    dir_name 为任务目录名，如 run_001_T400_100_Ge50_Si50。
    """
    temps = _ensure_list(total_config.get("substrate_temperature"))
    xyz_list = _ensure_list(total_config.get("initial_xyz"))
    species_list = _ensure_list(total_config.get("inject_species_weights"))

    base = {
        "Dirsstopsteps": total_config.get("Dirsstopsteps"),
        "inject_temperature": total_config.get("inject_temperature"),
        "inject_xy_spacing": total_config.get("inject_xy_spacing"),
        "injection_count": total_config.get("injection_count"),
        "z_spread": total_config.get("z_spread", 50.0),
        "local_surface_radius": total_config.get("local_surface_radius"),
        "surface_height_offset": total_config.get("surface_height_offset"),
        "cluster_cutoff": total_config.get("cluster_cutoff"),
        "remove_incident_particles": total_config.get("remove_incident_particles", True),
        "velocity_magnitude": total_config.get("velocity_magnitude"),
        "theta_sigma_deg": total_config.get("theta_sigma_deg", 5.0),
        "enable_d3": total_config.get("enable_d3", False),
        "time_step": total_config.get("time_step"),
        "run_steps": total_config.get("run_steps"),
        "box_z": total_config.get("box_z"),
        "gpumd_command": total_config.get("gpumd_command"),
        "initial_xyz_dir": total_config.get("initial_xyz_dir", "Substrate"),
        "substrate_replicate": total_config.get("substrate_replicate"),
    }

    tasks = []
    for idx, (T, xyz, species) in enumerate(
        itertools.product(temps, xyz_list, species_list), start=1
    ):
        config = dict(base)
        config["substrate_temperature"] = T
        config["initial_xyz"] = xyz if isinstance(xyz, str) else xyz
        config["inject_species_weights"] = _species_weights_to_dict(species)

        xyz_base = os.path.splitext(os.path.basename(config["initial_xyz"]))[0]
        species_name = "unknown"
        if isinstance(species, dict) and "name" in species:
            species_name = species["name"].replace(" ", "_")
        dir_name = f"run_{idx:03d}_T{T}_{xyz_base}_{species_name}"
        tasks.append((idx, config, dir_name))
    return tasks


def prepare_task_dirs(
    batch_root,
    total_config_path,
    demo_src_dir,
    gpu_list=None,
    source_root=None,
):
    """
    在 batch_root 下为所有任务创建目录并写入 config、拷贝文件。
    - source_root: 项目根目录，用于 initial_xyz_dir、Substrate、nep.txt 回退
    - gpu_list: 如 [0,1,2,3] 则按任务序号轮转 GPU
    """
    if source_root is None:
        source_root = batch_root

    total_config = load_total_config(total_config_path)
    tasks = expand_tasks(total_config)
    if gpu_list is None:
        gpu_list = total_config.get("gpu_list") or [0]

    substrate_dir = resolve_initial_xyz_dir(source_root, total_config)
    nep_src = os.path.join(demo_src_dir, "nep.txt")
    if not os.path.isfile(nep_src):
        nep_src = os.path.join(source_root, "nep.txt")

    for idx, config, dir_name in tasks:
        task_dir = os.path.join(batch_root, dir_name)
        os.makedirs(task_dir, exist_ok=True)

        config["gpu_device"] = gpu_list[(idx - 1) % len(gpu_list)]

        xyz_filename = os.path.basename(config["initial_xyz"])
        config["initial_xyz"] = xyz_filename

        config_path = os.path.join(task_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        if os.path.isfile(nep_src):
            shutil.copy2(nep_src, os.path.join(task_dir, "nep.txt"))

        xyz_src = os.path.join(substrate_dir, xyz_filename)
        if os.path.isfile(xyz_src):
            shutil.copy2(xyz_src, os.path.join(task_dir, xyz_filename))
        else:
            raise FileNotFoundError(
                f"衬底文件不存在: {xyz_src} "
                f"(initial_xyz_dir={config.get('initial_xyz_dir')})"
            )

        for name in ("demo.py", "utility.py"):
            src = os.path.join(demo_src_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(task_dir, name))

    return [dir_name for _, _, dir_name in tasks]


def list_task_dirs(batch_root):
    """返回 batch_root 下所有 run_ 开头的任务目录名，按名称排序。"""
    if not os.path.isdir(batch_root):
        return []
    names = [
        d
        for d in os.listdir(batch_root)
        if os.path.isdir(os.path.join(batch_root, d)) and d.startswith("run_")
    ]
    return sorted(names)
