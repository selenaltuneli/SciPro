import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pipeline_with_ui import COL_ATM, COL_DATE, COL_VAL, OUT_DIR, build_r_json_from_excel

WEEKLY_RE = re.compile(r"^weekly_(\d{8})_(\d{8})\.xlsx$")


def parse_week_dates(filename: str) -> tuple[datetime, datetime]:
    m = WEEKLY_RE.match(filename)
    if not m:
        raise ValueError(f"File name does not match weekly format: {filename}")
    start = datetime.strptime(m.group(1), "%Y%m%d")
    end = datetime.strptime(m.group(2), "%Y%m%d")
    return start, end


def run_one_week(
    weekly_file: Path,
    scenario: str,
    optimization_script: Path,
    optimizer_runner_script: Path,
    atm_master_file: Path,
    edges_file: Path,
    python_exec: str,
    solver_threads: int,
) -> dict[str, Any]:
    start_dt, end_dt = parse_week_dates(weekly_file.name)
    reopt_dt = start_dt - timedelta(days=1)
    prefix = f"{scenario}_end_{end_dt.strftime('%Y%m%d')}"

    build_r_json_from_excel(
        excel_path=str(weekly_file),
        out_dir=OUT_DIR,
        prefix=prefix,
        planning_start=start_dt,
        planning_end=end_dt,
        reopt_date=reopt_dt,
        col_date=COL_DATE,
        col_atm=COL_ATM,
        col_val=COL_VAL,
    )

    week_out_dir = Path(OUT_DIR) / prefix
    r_file = week_out_dir / f"{prefix}_Pred.json"
    meta_file = week_out_dir / f"{prefix}_meta.json"
    gurobi_logfile = week_out_dir / f"gurobi_{prefix}.log"

    cmd = [
        python_exec,
        str(optimizer_runner_script),
        "--optimization-script",
        str(optimization_script),
        "--atm-master-file",
        str(atm_master_file),
        "--edges-file",
        str(edges_file),
        "--r-file",
        str(r_file),
        "--meta-file",
        str(meta_file),
        "--threads",
        str(solver_threads),
        "--gurobi-logfile",
        str(gurobi_logfile),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Optimization script writes result txt next to r_file; read from meta for exact date labels.
    result_file = None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        s = meta["t_to_date"]["1"]
        e = meta["t_to_date"][str(meta["horizon_days"])]
        result_file = week_out_dir / f"optimization result {s} to {e}.txt"
    except Exception:
        pass

    return {
        "weekly_file": str(weekly_file),
        "prefix": prefix,
        "returncode": proc.returncode,
        "result_file": str(result_file) if result_file else "",
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-25:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-25:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline + optimization for all weekly Excel files.")
    parser.add_argument("--weekly-dir", default="weekly_excels")
    parser.add_argument("--scenario", default="Scenario1")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--solver-threads", type=int, default=1, help="Gurobi threads per worker process")
    parser.add_argument(
        "--start-after",
        default="",
        help="Skip files up to and including this weekly filename (e.g. weekly_20070226_20070304.xlsx)",
    )
    parser.add_argument("--optimization-script", default="Optimization Final.py")
    parser.add_argument("--optimizer-runner-script", default="optimize_noninteractive.py")
    parser.add_argument("--atm-master-file", default="outputs/weeklyavg_FIXED_end_20070225_atm_master.csv")
    parser.add_argument("--edges-file", default="outputs/weeklyavg_FIXED_end_20070225_edges_dist_km.csv")
    parser.add_argument("--summary-file", default="outputs/batch_summary.json")
    args = parser.parse_args()

    weekly_dir = Path(args.weekly_dir)
    optimization_script = Path(args.optimization_script)
    optimizer_runner_script = Path(args.optimizer_runner_script)
    atm_master_file = Path(args.atm_master_file)
    edges_file = Path(args.edges_file)

    if not weekly_dir.exists():
        raise FileNotFoundError(f"weekly dir not found: {weekly_dir}")
    if not optimization_script.exists():
        raise FileNotFoundError(f"optimization script not found: {optimization_script}")
    if not optimizer_runner_script.exists():
        raise FileNotFoundError(f"optimizer runner script not found: {optimizer_runner_script}")
    if not atm_master_file.exists():
        raise FileNotFoundError(f"atm master file not found: {atm_master_file}")
    if not edges_file.exists():
        raise FileNotFoundError(f"edges file not found: {edges_file}")

    weekly_files = sorted([p for p in weekly_dir.glob("weekly_*.xlsx") if WEEKLY_RE.match(p.name)])
    if not weekly_files:
        raise RuntimeError(f"No weekly files found in {weekly_dir}")

    if args.start_after:
        weekly_files = [p for p in weekly_files if p.name > args.start_after]
        if not weekly_files:
            raise RuntimeError(f"No weekly files found after {args.start_after}")

    print(f"Weekly files found: {len(weekly_files)}")
    print(f"Workers: {args.workers}")
    print(f"Solver threads per worker: {args.solver_threads}")
    print(f"Scenario: {args.scenario}")

    results: list[dict[str, Any]] = []
    py_exec = sys.executable

    if args.workers <= 1:
        for wf in weekly_files:
            print(f"\n[RUN] {wf.name}")
            res = run_one_week(
                weekly_file=wf,
                scenario=args.scenario,
                optimization_script=optimization_script,
                optimizer_runner_script=optimizer_runner_script,
                atm_master_file=atm_master_file,
                edges_file=edges_file,
                python_exec=py_exec,
                solver_threads=args.solver_threads,
            )
            results.append(res)
            print(f"[DONE] {wf.name} rc={res['returncode']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_map = {
                ex.submit(
                    run_one_week,
                    wf,
                    args.scenario,
                    optimization_script,
                    optimizer_runner_script,
                    atm_master_file,
                    edges_file,
                    py_exec,
                    args.solver_threads,
                ): wf
                for wf in weekly_files
            }
            for fut in as_completed(future_map):
                wf = future_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {
                        "weekly_file": str(wf),
                        "prefix": "",
                        "returncode": 999,
                        "result_file": "",
                        "stdout_tail": "",
                        "stderr_tail": str(e),
                    }
                results.append(res)
                print(f"[DONE] {wf.name} rc={res['returncode']}")

    Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_file).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = sum(1 for r in results if r["returncode"] == 0)
    fail = len(results) - ok
    print("\n=== BATCH SUMMARY ===")
    print(f"Success: {ok}")
    print(f"Failed : {fail}")
    print(f"Summary file: {args.summary_file}")


if __name__ == "__main__":
    main()
