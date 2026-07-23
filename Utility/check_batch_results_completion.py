"""
遍历 `batch_results` 下的 `run_*` 批组目录，检查“计算是否真正完成”。

完成判定（面向你这个目录结构）：
1) 读取 `run_*/config.yaml`，取 `expected_rounds = Dirsstopsteps`
2) 只检查 `run_*/<数字>/dump.xyz` 是否存在以及是否非空（size > empty_threshold）
3) 只有当：
   - 数字子目录数 == expected_rounds
   - 丢失 dump == 0
   - 空 dump == 0
   才认为该 batch group 完成（status=done）

输出：
- 屏幕打印一个表格（markdown 格式）
- 额外列出所有 done 的 run 目录路径
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple

import yaml


SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_ROOT)


def _safe_load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except FileNotFoundError:
        return {}


def _parse_species_label_from_run_name(run_name: str) -> Optional[str]:
    """
    run_001_T200_100_Ge50_Si50 -> Ge50_Si50
    """
    parts = run_name.split("_")
    if len(parts) >= 6 and parts[-2].startswith("Ge") and parts[-1].startswith("Si"):
        return f"{parts[-2]}_{parts[-1]}"
    return None


def _round_to_pct(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        fv = float(v)
        return int(round(fv * 100))
    except Exception:
        return None


def _format_weights_ge_si(cfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    weights = cfg.get("inject_species_weights")
    if not isinstance(weights, dict):
        return None, None
    ge = _round_to_pct(weights.get("Ge"))
    si = _round_to_pct(weights.get("Si"))
    return ge, si


def _expected_rounds_from_cfg(cfg: Dict[str, Any]) -> Optional[int]:
    # total_config.yaml 使用 Dirsstopsteps
    for k in ("Dirsstopsteps", "DirsStopSteps", "dirsstopsteps", "run_steps"):
        if k in cfg:
            v = cfg.get(k)
            if v is None:
                continue
            try:
                return int(v)
            except Exception:
                continue
    return None


def _count_dump_in_numeric_subdirs(
    run_dir: str,
    dump_filename: str,
    empty_threshold: int,
) -> Tuple[int, int, int]:
    """
    只检查 run_dir 下面“第一层数字子目录”的 dump 文件：
    run_dir/<i>/<dump_filename>

    返回：
      (digit_dir_count, missing_dump_count, empty_dump_count)
    """
    digit_dir_count = 0
    missing_dump_count = 0
    empty_dump_count = 0

    with os.scandir(run_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            name = entry.name
            if not name.isdigit():
                continue

            digit_dir_count += 1
            dump_path = os.path.join(run_dir, name, dump_filename)

            if not os.path.isfile(dump_path):
                missing_dump_count += 1
                continue

            try:
                size = os.path.getsize(dump_path)
            except Exception:
                missing_dump_count += 1
                continue

            if size <= empty_threshold:
                empty_dump_count += 1

    return digit_dir_count, missing_dump_count, empty_dump_count


def scan_one_run(
    batch_results_root: str,
    run_name: str,
    dump_filename: str,
    empty_threshold: int,
) -> Dict[str, Any]:
    run_dir = os.path.join(batch_results_root, run_name)
    cfg_path = os.path.join(run_dir, "config.yaml")
    cfg = _safe_load_yaml(cfg_path)
    expected_rounds = _expected_rounds_from_cfg(cfg)

    digit_dir_count, missing_dump_count, empty_dump_count = _count_dump_in_numeric_subdirs(
        run_dir=run_dir,
        dump_filename=dump_filename,
        empty_threshold=empty_threshold,
    )

    substrate_temperature = cfg.get("substrate_temperature")
    initial_xyz = cfg.get("initial_xyz")
    species_label = _parse_species_label_from_run_name(run_name)
    ge_pct, si_pct = _format_weights_ge_si(cfg)

    status: str
    reason: str

    if not expected_rounds:
        status = "unknown_config"
        reason = "Missing Dirsstopsteps in config.yaml (or config read failed)"
    else:
        if digit_dir_count != expected_rounds:
            status = "incomplete"
            reason = f"digit_dirs {digit_dir_count} != expected {expected_rounds}"
        elif missing_dump_count != 0:
            status = "incomplete"
            reason = f"missing dump.xyz: {missing_dump_count}"
        elif empty_dump_count != 0:
            status = "incomplete"
            reason = f"empty dump.xyz: {empty_dump_count} (size <= {empty_threshold})"
        else:
            status = "done"
            reason = "dump.xyz complete and non-empty"

    return {
        "run_dir": run_dir,
        "run_name": run_name,
        "substrate_temperature": substrate_temperature,
        "initial_xyz": initial_xyz,
        "species_label": species_label,
        "ge_pct": ge_pct,
        "si_pct": si_pct,
        "expected_rounds": expected_rounds,
        "digit_dir_count": digit_dir_count,
        "missing_dump_count": missing_dump_count,
        "empty_dump_count": empty_dump_count,
        "status": status,
        "reason": reason,
    }


def _md_table(rows: list[Dict[str, Any]]) -> str:
    headers = [
        "run_name",
        "T(K)",
        "initial_xyz",
        "species",
        "expected",
        "digit_dirs",
        "missing",
        "empty",
        "status",
    ]

    def cell(r: Dict[str, Any], h: str) -> str:
        v = r.get(h)
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def species_cell(r: Dict[str, Any]) -> str:
        if r.get("species_label"):
            return str(r["species_label"])
        ge_pct = r.get("ge_pct")
        si_pct = r.get("si_pct")
        if ge_pct is None and si_pct is None:
            return ""
        return f"Ge{ge_pct}_Si{si_pct}"

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in rows:
        vals = []
        for h in headers:
            if h == "species":
                vals.append(species_cell(r))
            elif h == "T(K)":
                vals.append(cell(r, "substrate_temperature"))
            elif h == "expected":
                vals.append(cell(r, "expected_rounds"))
            elif h == "digit_dirs":
                vals.append(cell(r, "digit_dir_count"))
            elif h == "missing":
                vals.append(cell(r, "missing_dump_count"))
            elif h == "empty":
                vals.append(cell(r, "empty_dump_count"))
            else:
                vals.append(cell(r, h))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 batch_results 中各 run_* 是否真正完成")
    parser.add_argument(
        "--batch-results",
        default=os.path.join(PROJECT_ROOT, "batch_results"),
        help="batch_results 根目录路径（默认：项目根目录下的 batch_results）",
    )
    parser.add_argument("--dump-filename", default="dump.xyz", help="用于判定完成与否的输出文件名")
    parser.add_argument(
        "--empty-threshold",
        type=int,
        default=0,
        help="将 dump.xyz 大小 <= empty_threshold 视为“空/未完成”（默认 0）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行 worker 数量（默认：CPU 核心数，且不会超过 run 数量）",
    )
    parser.add_argument("--limit", type=int, default=0, help="只扫描前 N 个 run_*（默认：全部）")
    args = parser.parse_args()

    # Avoid garbled CJK output in some terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    batch_results_root = os.path.abspath(args.batch_results)
    if not os.path.isdir(batch_results_root):
        raise NotADirectoryError(f"batch_results 不存在或不是目录: {batch_results_root}")

    run_names: list[str] = []
    with os.scandir(batch_results_root) as it:
        for entry in it:
            if entry.is_dir() and entry.name.startswith("run_"):
                run_names.append(entry.name)
    run_names = sorted(run_names)

    if args.limit and args.limit > 0:
        run_names = run_names[: args.limit]

    if not run_names:
        print(f"No run_* directories found: {batch_results_root}")
        return

    max_workers = args.workers
    if max_workers <= 0:
        max_workers = os.cpu_count() or 4
    max_workers = min(max_workers, len(run_names))

    print(f"batch_results: {batch_results_root}")
    print(f"run count: {len(run_names)}")
    print(f"workers: {max_workers} (parallel scan per run)")
    print("")

    results: list[Dict[str, Any]] = []

    # 单 run / 单 worker 时走顺序，避免进程池的启动开销
    if len(run_names) == 1 or max_workers <= 1:
        for rn in run_names:
            results.append(
                scan_one_run(
                    batch_results_root=batch_results_root,
                    run_name=rn,
                    dump_filename=args.dump_filename,
                    empty_threshold=args.empty_threshold,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    scan_one_run,
                    batch_results_root,
                    rn,
                    args.dump_filename,
                    args.empty_threshold,
                ): rn
                for rn in run_names
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    rn = futures[fut]
                    results.append(
                        {
                            "run_dir": os.path.join(batch_results_root, rn),
                            "run_name": rn,
                            "status": "error",
                            "reason": str(e),
                        }
                    )

    results.sort(key=lambda x: str(x.get("run_name", "")))

    print(_md_table(results))

    done_rows = [r for r in results if r.get("status") == "done"]
    print("")
    if done_rows:
        print("Done directories (dump.xyz complete and non-empty):")
        for r in done_rows:
            print(r["run_dir"])
    else:
        print("No completed runs detected (may still be running, or has empty/missing dump.xyz).")

    not_done = [r for r in results if r.get("status") != "done"]
    if not_done:
        print("")
        print("Not done reason summary (one run per line):")
        for r in not_done:
            rn = r.get("run_name", "")
            st = r.get("status", "")
            reason = r.get("reason", "")
            if reason:
                print(f"{rn}: {st} - {reason}")
            else:
                print(f"{rn}: {st}")


if __name__ == "__main__":
    main()

