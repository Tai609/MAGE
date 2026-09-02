import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_explicit_paper_ids(raw_values: Optional[List[str]]) -> List[int]:
    ids: List[int] = []
    if not raw_values:
        return ids

    for raw in raw_values:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid paper id '{part}'. Use integers such as 12 20 22 or 12,20,22."
                ) from exc
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run extract_main.py sequentially for paper_<index> directories. "
            "Supports either a contiguous range (--start/--end) or an explicit "
            "paper id list (--paper-ids)."
        )
    )
    parser.add_argument("--start", type=int, help="Start paper index, e.g. 10")
    parser.add_argument("--end", type=int, help="End paper index, e.g. 11")
    parser.add_argument(
        "--paper-ids",
        nargs="+",
        help=(
            "Explicit paper ids to run. Supports space- or comma-separated values, "
            "e.g. --paper-ids 12 20 22 or --paper-ids 12,20,22."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root where extract_main.py is located",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed_papers"),
        help="Base directory that contains paper_<index> folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output_extract"),
        help="Base directory to write output_<index> folders",
    )
    parser.add_argument(
        "--output-name-template",
        default="output_{index}",
        help=(
            "Output folder naming template under --output-root. "
            "Available fields: {index}, {paper_index}, {paper}, {output}. "
            "Example: paper_{index}"
        ),
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=Path("prompts/features_to_extract_HER.txt"),
        help="Feature file passed to extract_main.py",
    )
    parser.add_argument(
        "--mode",
        default="extract",
        choices=["extract", "both", "generate-ml-only"],
        help="Mode passed to extract_main.py (default: extract)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Optional --processes value forwarded to extract_main.py",
    )
    parser.add_argument(
        "--text-model",
        default="",
        help="Optional --text-model forwarded to extract_main.py",
    )
    parser.add_argument(
        "--vision-model",
        default="",
        help="Optional --vision-model forwarded to extract_main.py",
    )
    parser.add_argument(
        "--alignment-model",
        default="",
        help="Optional --alignment-model forwarded to extract_main.py",
    )
    parser.add_argument(
        "--ml-model",
        default="",
        help="Optional --ml-model forwarded to extract_main.py",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for running extract_main.py",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one paper fails (default: continue)",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Repeat start..end forever until manually interrupted (Ctrl+C)",
    )
    return parser.parse_args()


def to_abs(project_root: Path, maybe_relative: Path) -> Path:
    if maybe_relative.is_absolute():
        return maybe_relative
    return (project_root / maybe_relative).resolve()


def run_one(
    python_exec: str,
    project_root: Path,
    input_path: Path,
    output_dir: Path,
    mode: str,
    feature_file: Path,
    processes: Optional[int],
    text_model: str,
    vision_model: str,
    alignment_model: str,
    ml_model: str,
) -> Dict[str, Any]:
    cmd = [
        python_exec,
        "extract_main.py",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--mode",
        mode,
        "--feature-file",
        str(feature_file),
    ]
    if processes is not None:
        cmd.extend(["--processes", str(processes)])
    if text_model.strip():
        cmd.extend(["--text-model", text_model.strip()])
    if vision_model.strip():
        cmd.extend(["--vision-model", vision_model.strip()])
    if alignment_model.strip():
        cmd.extend(["--alignment-model", alignment_model.strip()])
    if ml_model.strip():
        cmd.extend(["--ml-model", ml_model.strip()])

    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[{now()}] Running: {printable}")
    started_at = datetime.now()
    started_perf = time.perf_counter()
    result = subprocess.run(cmd, cwd=project_root, env=os.environ.copy())
    elapsed_seconds = time.perf_counter() - started_perf
    finished_at = datetime.now()
    return {
        "return_code": result.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed_seconds,
        "requested_text_model": text_model.strip() or None,
        "requested_vision_model": vision_model.strip() or None,
        "requested_alignment_model": alignment_model.strip() or None,
        "requested_ml_model": ml_model.strip() or None,
    }


def find_result_metadata_file(output_dir: Path) -> Optional[Path]:
    metadata_dir = output_dir / "metadata"
    if not metadata_dir.exists():
        return None

    preferred = metadata_dir / "full_result.json"
    if preferred.exists():
        return preferred

    candidates = [p for p in metadata_dir.glob("*_result.json") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_json_file(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def write_per_paper_log(
    output_root: Path,
    paper_index: int,
    input_path: Path,
    output_dir: Path,
    run_info: Dict[str, Any],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = output_root / f"paper_{paper_index}_extract_log_{timestamp}.json"

    metadata_file = find_result_metadata_file(output_dir)
    metadata = read_json_file(metadata_file) if metadata_file else {}

    log_data: Dict[str, Any] = {
        "paper_index": paper_index,
        "paper_id": f"paper_{paper_index}",
        "status": "success" if run_info.get("return_code", 1) == 0 else "failed",
        "return_code": run_info.get("return_code"),
        "started_at": run_info.get("started_at").isoformat(timespec="seconds") if run_info.get("started_at") else None,
        "finished_at": run_info.get("finished_at").isoformat(timespec="seconds") if run_info.get("finished_at") else None,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "metadata_file": str(metadata_file) if metadata_file else None,
        "text_extraction_model": metadata.get("text_extraction_model") or metadata.get("model_name"),
        "text_extraction_time_sec": metadata.get("text_extraction_duration_sec"),
        "image_extraction_model": metadata.get("image_extraction_model"),
        "image_extraction_time_sec": metadata.get("image_extraction_duration_sec"),
        "alignment_model": metadata.get("alignment_model"),
        "alignment_time_sec": metadata.get("alignment_duration_sec"),
        "requested_text_model": run_info.get("requested_text_model"),
        "requested_vision_model": run_info.get("requested_vision_model"),
        "requested_alignment_model": run_info.get("requested_alignment_model"),
        "requested_ml_model": run_info.get("requested_ml_model"),
        "total_time_sec": round(float(run_info.get("elapsed_seconds", 0.0)), 3),
        "total_extraction_time_sec": metadata.get("total_extraction_duration_sec"),
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    return log_path


def main() -> int:
    args = parse_args()

    explicit_paper_ids = parse_explicit_paper_ids(args.paper_ids)
    if explicit_paper_ids:
        paper_indices = explicit_paper_ids
    else:
        if args.start is None or args.end is None:
            print(f"[{now()}] Error: provide either --paper-ids or both --start and --end.")
            return 2
        if args.start > args.end:
            print(f"[{now()}] Error: --start ({args.start}) cannot be greater than --end ({args.end}).")
            return 2
        paper_indices = list(range(args.start, args.end + 1))

    project_root = args.project_root.resolve()
    data_root = to_abs(project_root, args.data_root)
    output_root = to_abs(project_root, args.output_root)
    feature_file = to_abs(project_root, args.feature_file)

    if not (project_root / "extract_main.py").exists():
        print(f"[{now()}] Error: extract_main.py not found under {project_root}")
        return 2

    if not data_root.exists():
        print(f"[{now()}] Error: data root does not exist: {data_root}")
        return 2

    if not feature_file.exists():
        print(f"[{now()}] Error: feature file does not exist: {feature_file}")
        return 2

    output_root.mkdir(parents=True, exist_ok=True)

    cycle = 0
    while True:
        cycle += 1
        print(f"\n[{now()}] ===== Batch cycle {cycle} started =====")

        failed_indices: List[int] = []
        for idx in paper_indices:
            input_path = data_root / f"paper_{idx}"
            try:
                output_name = args.output_name_template.format(
                    index=idx,
                    paper_index=idx,
                    paper=f"paper_{idx}",
                    output=f"output_{idx}",
                )
            except KeyError as e:
                print(
                    f"[{now()}] Error: invalid --output-name-template field {e}. "
                    "Use only {index}, {paper_index}, {paper}, {output}."
                )
                return 2
            output_dir = output_root / output_name

            if not input_path.exists():
                print(f"[{now()}] Skip paper_{idx}: input path not found -> {input_path}")
                failed_indices.append(idx)
                if args.stop_on_error:
                    print(f"[{now()}] Stopping because --stop-on-error is enabled.")
                    return 1
                continue

            run_info = run_one(
                python_exec=args.python,
                project_root=project_root,
                input_path=input_path,
                output_dir=output_dir,
                mode=args.mode,
                feature_file=feature_file,
                processes=args.processes,
                text_model=args.text_model,
                vision_model=args.vision_model,
                alignment_model=args.alignment_model,
                ml_model=args.ml_model,
            )
            return_code = int(run_info.get("return_code", 1))
            log_path = write_per_paper_log(
                output_root=output_root,
                paper_index=idx,
                input_path=input_path,
                output_dir=output_dir,
                run_info=run_info,
            )
            print(f"[{now()}] Saved paper log: {log_path}")

            if return_code == 0:
                print(f"[{now()}] paper_{idx} finished successfully.")
            else:
                print(f"[{now()}] paper_{idx} failed with exit code {return_code}.")
                failed_indices.append(idx)
                if args.stop_on_error:
                    print(f"[{now()}] Stopping because --stop-on-error is enabled.")
                    return return_code

        if failed_indices:
            joined = ", ".join(f"paper_{i}" for i in failed_indices)
            print(f"[{now()}] Cycle {cycle} completed with failures: {joined}")
        else:
            print(f"[{now()}] Cycle {cycle} completed successfully for all papers.")

        if not args.repeat:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
