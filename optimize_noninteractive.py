import argparse
from pathlib import Path


def build_selected_files_block(atm_master_file: str, edges_file: str, r_file: str, meta_file: str) -> str:
    return (
        "selected_files = {\n"
        f"    'atm_master_file': {atm_master_file!r},\n"
        f"    'edges_file': {edges_file!r},\n"
        f"    'r_file': {r_file!r},\n"
        f"    'meta_file': {meta_file!r},\n"
        "}"
    )


def run_optimization(
    optimization_script: Path,
    atm_master_file: str,
    edges_file: str,
    r_file: str,
    meta_file: str,
    threads: int,
    gurobi_logfile: str,
) -> None:
    code = optimization_script.read_text(encoding="utf-8")

    selected_files_block = build_selected_files_block(
        atm_master_file=atm_master_file,
        edges_file=edges_file,
        r_file=r_file,
        meta_file=meta_file,
    )

    if "selected_files = ask_user_for_input_files()" not in code:
        raise RuntimeError("Could not find file-selection line in Optimization Final.py")
    code = code.replace("selected_files = ask_user_for_input_files()", selected_files_block, 1)

    # Ensure per-run log file (avoid collisions across parallel workers).
    code = code.replace('GUROBI_LOGFILE = "gurobi.log"', f"GUROBI_LOGFILE = {gurobi_logfile!r}", 1)

    # Force deterministic/safe per-process thread usage.
    if "model.Params.LazyConstraints = 1" in code:
        code = code.replace(
            "model.Params.LazyConstraints = 1",
            f"model.Params.LazyConstraints = 1\nmodel.Params.Threads = {int(threads)}",
            1,
        )

    # Keep execution alive after solve (there is a known typo in final breakdown).
    code = code.replace(
        "total_val         = inv_co",
        "total_val         = inv_cost_val + stockout_cost_val + route_cost_val",
        1,
    )

    ns = {"__name__": "__main__", "__file__": str(optimization_script)}
    exec(compile(code, str(optimization_script), "exec"), ns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optimization Final.py without UI file pickers.")
    parser.add_argument("--optimization-script", default="Optimization Final.py")
    parser.add_argument("--atm-master-file", required=True)
    parser.add_argument("--edges-file", required=True)
    parser.add_argument("--r-file", required=True)
    parser.add_argument("--meta-file", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gurobi-logfile", default="")
    args = parser.parse_args()

    log_file = args.gurobi_logfile
    if not log_file:
        r_path = Path(args.r_file)
        log_file = str(r_path.parent / f"gurobi_{r_path.stem}.log")

    run_optimization(
        optimization_script=Path(args.optimization_script),
        atm_master_file=args.atm_master_file,
        edges_file=args.edges_file,
        r_file=args.r_file,
        meta_file=args.meta_file,
        threads=args.threads,
        gurobi_logfile=log_file,
    )


if __name__ == "__main__":
    main()
