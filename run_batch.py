"""
批量并行执行入口：按 total_config.yaml 展开任务目录后，
多个大任务并行跑（每 GPU 一个），某 GPU 上的任务结束后立即在该 GPU 上启动下一任务。
支持断点续跑（已完成的任务目录记录在 .batch_done 中并跳过）。
"""
import argparse
import os
import queue
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from batch_utils import (
    list_task_dirs,
    load_total_config,
    prepare_task_dirs,
)

# 本脚本所在目录，同时也是 total_config.yaml 的来源目录
SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
TOTAL_CONFIG_PATH = os.path.join(SCRIPT_ROOT, "total_config.yaml")
# 默认从 SingleRunInputScript 读取 demo.py/utility.py/nep.txt
DEMO_SRC_DIR = os.path.join(SCRIPT_ROOT, "SingleRunInputScript")


def resolve_tasks_root(total_config):
    """
    根据 total_config 的 output_dir 决定 run_XXX 目录的落点。
    - output_dir 未配置/为空：保持旧行为（落在 SCRIPT_ROOT）
    - 相对路径：相对 SCRIPT_ROOT
    - 绝对路径：直接使用
    """
    output_dir = total_config.get("output_dir")
    if not output_dir:
        return SCRIPT_ROOT
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(SCRIPT_ROOT, output_dir)


def read_done_set(done_file):
    """读取已完成任务目录名集合。"""
    if not os.path.isfile(done_file):
        return set()
    with open(done_file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def mark_done(task_dir_name, done_file):
    """将任务目录名追加到 .batch_done。"""
    with open(done_file, "a", encoding="utf-8") as f:
        f.write(task_dir_name + "\n")


def run_one_task(task_dir_name, gpu_queue, tasks_root):
    """
    在指定任务目录内执行 demo.py，使用从 gpu_queue 取出的 GPU。
    Unix：nohup 后台跑，标准输出/错误写入任务目录下 nohup.log，shell wait 取得真实退出码。
    Windows：无前述工具，仍直接 subprocess.run。
    执行完毕后将 GPU 放回队列。返回 (task_dir_name, returncode)。
    """
    task_path = os.path.join(tasks_root, task_dir_name)
    demo_py = os.path.join(task_path, "demo.py")
    log_path = os.path.join(task_path, "nohup.log")
    gpu_id = None
    try:
        gpu_id = gpu_queue.get()
        env = {**os.environ, "BATCH_GPU_ID": str(gpu_id)}
        if sys.platform == "win32":
            ret = subprocess.run(
                [sys.executable, "demo.py"],
                cwd=task_path,
                env=env,
            )
        else:
            shell_cmd = (
                f"nohup {shlex.quote(sys.executable)} demo.py "
                f"> {shlex.quote(log_path)} 2>&1 & pid=$!; wait $pid"
            )
            ret = subprocess.run(
                shell_cmd,
                cwd=task_path,
                env=env,
                shell=True,
                executable="/bin/sh",
            )
        return (task_dir_name, ret.returncode)
    finally:
        if gpu_id is not None:
            gpu_queue.put(gpu_id)


def main():
    parser = argparse.ArgumentParser(description="批量并行执行 demo.py 任务（每 GPU 一大任务，跑完即上新任务）")
    parser.add_argument(
        "--config",
        default=TOTAL_CONFIG_PATH,
        help="total_config.yaml 路径（默认项目根目录 total_config.yaml）",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="仅生成任务目录与 config，不执行任务",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略 .batch_done，从头执行所有任务",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    batch_start = time.time()

    # 1. 若尚无任务目录，则先准备
    total_config = load_total_config(config_path)
    tasks_root = resolve_tasks_root(total_config)
    done_file = os.path.join(tasks_root, ".batch_done")

    task_dirs = list_task_dirs(tasks_root)
    if not task_dirs:
        if not os.path.isdir(DEMO_SRC_DIR):
            print(f"未找到 demo 源目录 {DEMO_SRC_DIR}，退出")
            sys.exit(1)
        gpu_list = total_config.get("gpu_list")  # 可选，如 [0,1,2,3] 轮转
        task_dirs = prepare_task_dirs(
            tasks_root,
            config_path,
            DEMO_SRC_DIR,
            gpu_list=gpu_list,
            source_root=SCRIPT_ROOT,
        )
        print(f"已生成 {len(task_dirs)} 个任务目录")
        if args.prepare_only:
            return

    if args.prepare_only:
        return

    done = set() if args.no_resume else read_done_set(done_file)
    to_run = [d for d in task_dirs if d not in done]
    if not to_run:
        print("所有任务均已完成，无需执行")
        return
    print(f"待执行任务数: {len(to_run)} / {len(task_dirs)}")

    # 读取 gpu_list；若无则退化为单卡，等价于串行
    gpu_list = total_config.get("gpu_list")
    if not gpu_list:
        gpu_list = [total_config.get("gpu_device", 0)]
    gpu_queue = queue.Queue()
    for gid in gpu_list:
        gpu_queue.put(gid)
    print(f"并行 GPU 数: {len(gpu_list)}")

    # 过滤掉无 demo.py 的任务
    to_run = [
        d
        for d in to_run
        if os.path.isfile(os.path.join(tasks_root, d, "demo.py"))
    ]
    if not to_run:
        print("无有效任务可执行")
        return

    # 并行执行：每任务从 GPU 队列取一块 GPU，跑完归还
    failed = []
    with ThreadPoolExecutor(max_workers=len(gpu_list)) as executor:
        futures = {
            executor.submit(run_one_task, task_dir_name, gpu_queue, tasks_root): task_dir_name
            for task_dir_name in to_run
        }
        for i, future in enumerate(as_completed(futures), start=1):
            task_dir_name = futures[future]
            try:
                _, returncode = future.result()
                if returncode == 0:
                    mark_done(task_dir_name, done_file)
                    print(f"[{i}/{len(to_run)}] 已完成: {task_dir_name}")
                else:
                    failed.append((task_dir_name, returncode))
                    print(f"[{i}/{len(to_run)}] 失败(退出码 {returncode}): {task_dir_name}")
            except Exception as e:
                failed.append((task_dir_name, str(e)))
                print(f"[{i}/{len(to_run)}] 异常: {task_dir_name} -> {e}")

    batch_elapsed = time.time() - batch_start
    timing_log = os.path.join(tasks_root, "batch_scheduler_timing.log")
    with open(timing_log, "a", encoding="utf-8") as f:
        f.write(
            f"config={config_path}\n"
            f"tasks_root={tasks_root}\n"
            f"total_wall_sec={batch_elapsed:.1f}\n"
            f"total_wall_h={batch_elapsed / 3600:.4f}\n"
            f"finished_tasks={len(to_run) - len(failed)}/{len(to_run)}\n"
            f"failed={len(failed)}\n"
            f"---\n"
        )
    print(
        f"批量调度总墙钟时长: {batch_elapsed:.1f} s ({batch_elapsed / 3600:.2f} h) "
        f"-> {timing_log}"
    )

    if failed:
        print(f"共 {len(failed)} 个任务失败，其余已完成。")
        sys.exit(1)
    print("全部任务执行完毕。")


if __name__ == "__main__":
    main()
