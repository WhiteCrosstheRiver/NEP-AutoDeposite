"""
读取 total_config.yaml，统计 output_dir 下所有 run_XXX 工作目录的完成情况。

用法：
    python Utility/check_batch_status.py
    python Utility/check_batch_status.py --config total_config.yaml
"""
import argparse
import os
from typing import Dict, List, Tuple

import yaml


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_tasks_root(project_root: str, total_config: Dict) -> str:
    output_dir = total_config.get("output_dir")
    if not output_dir:
        return project_root
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(project_root, output_dir)


def list_run_dirs(tasks_root: str) -> List[str]:
    if not os.path.isdir(tasks_root):
        return []
    names = []
    for name in os.listdir(tasks_root):
        path = os.path.join(tasks_root, name)
        if os.path.isdir(path) and name.startswith("run_"):
            names.append(name)
    return sorted(names)


def read_done_set(done_file: str) -> set:
    if not os.path.isfile(done_file):
        return set()
    with open(done_file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def infer_task_state(task_path: str, done_set: set, task_name: str) -> Tuple[str, str]:
    if task_name in done_set:
        return "done", "记录于 .batch_done"

    dump_xyz = os.path.join(task_path, "dump.xyz")
    nohup_log = os.path.join(task_path, "nohup.log")
    has_dump = os.path.isfile(dump_xyz)
    has_log = os.path.isfile(nohup_log)

    if has_dump:
        return "done_like", "存在 dump.xyz（但未记录到 .batch_done）"
    if has_log:
        return "running_or_failed", "存在 nohup.log（可能在运行或失败）"
    return "pending", "未见输出文件"


def main() -> None:
    parser = argparse.ArgumentParser(description="统计批量任务完成情况")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "total_config.yaml"),
        help="total_config.yaml 路径（默认项目根目录）",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    project_root = os.path.dirname(config_path)
    total_config = load_yaml(config_path)
    tasks_root = resolve_tasks_root(project_root, total_config)

    run_dirs = list_run_dirs(tasks_root)
    done_file = os.path.join(tasks_root, ".batch_done")
    done_set = read_done_set(done_file)

    print(f"配置文件: {config_path}")
    print(f"任务目录: {tasks_root}")
    print(f"任务总数: {len(run_dirs)}")
    print(f".batch_done: {'存在' if os.path.isfile(done_file) else '不存在'}")
    print("-" * 72)

    counters = {
        "done": 0,
        "done_like": 0,
        "running_or_failed": 0,
        "pending": 0,
    }

    for name in run_dirs:
        task_path = os.path.join(tasks_root, name)
        state, reason = infer_task_state(task_path, done_set, name)
        counters[state] += 1
        print(f"{name:<45} {state:<18} {reason}")

    print("-" * 72)
    print(
        "汇总: "
        f"done={counters['done']}, "
        f"done_like={counters['done_like']}, "
        f"running_or_failed={counters['running_or_failed']}, "
        f"pending={counters['pending']}"
    )


if __name__ == "__main__":
    main()
