import os
import json
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime


# =========================
# SETTINGS
# =========================
OUT_DIR = "outputs"

COL_DATE = "FORECAST_DATE"
COL_ATM  = "CASHP_ID_ATM"
COL_VAL  = "Y_PRED_WITHDRWLS_ATM"


# =========================
# PIPELINE FUNCTION
# =========================
def build_r_json_from_excel(
    excel_path,
    out_dir,
    prefix,
    planning_start,
    planning_end,
    reopt_date,
    col_date,
    col_atm,
    col_val
):
    # create folder: outputs/<prefix>/
    scenario_out_dir = os.path.join(out_dir, prefix)
    os.makedirs(scenario_out_dir, exist_ok=True)

    df = pd.read_excel(excel_path)

    # basic typing
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df[col_atm] = df[col_atm].astype(str)
    df[col_val] = pd.to_numeric(df[col_val], errors="coerce")

    # drop unusable rows
    df = df.dropna(subset=[col_date, col_atm, col_val]).copy()

    # normalize dates to avoid time-component issues
    planning_start = pd.Timestamp(planning_start).normalize()
    planning_end = pd.Timestamp(planning_end).normalize()
    reopt_date = pd.Timestamp(reopt_date).normalize()

    df["_date_norm"] = df[col_date].dt.normalize()

    # filter planning window
    df = df[
        (df["_date_norm"] >= planning_start) &
        (df["_date_norm"] <= planning_end)
    ].copy()

    if df.empty:
        raise ValueError("No data found in the selected planning window.")

    # create t mapping from sorted dates
    unique_dates = sorted(df["_date_norm"].unique())

    date_to_t = {pd.Timestamp(d): i + 1 for i, d in enumerate(unique_dates)}
    t_to_date = {str(i + 1): pd.Timestamp(d).strftime("%d.%m.%Y") for i, d in enumerate(unique_dates)}

    df["_t"] = df["_date_norm"].map(date_to_t)

    # duplicate check
    dup = df.duplicated(subset=[col_atm, "_t"], keep=False)
    if dup.any():
        bad = df.loc[dup, [col_atm, col_date, "_t", col_val]].sort_values([col_atm, "_t"])
        raise ValueError(
            "Duplicate rows detected for the same (ATM, t).\n\n"
            f"Examples:\n{bad.head(20).to_string(index=False)}"
        )

    # build json: "atm|t" -> value
    r_json = {
        f"{row[col_atm]}|{int(row['_t'])}": float(row[col_val])
        for _, row in df.iterrows()
    }

    r_path = os.path.join(scenario_out_dir, f"{prefix}_Pred.json")
    with open(r_path, "w", encoding="utf-8") as f:
        json.dump(r_json, f, indent=2)

    # meta
    meta = {
        "scenario_prefix": prefix,
        "planning_start_date": planning_start.strftime("%d.%m.%Y"),
        "planning_end_date": planning_end.strftime("%d.%m.%Y"),
        "reoptimization_date": reopt_date.strftime("%d.%m.%Y"),
        "horizon_days": len(unique_dates),
        "t_index_start": 1,
        "t_to_date": t_to_date,
        "source_file": os.path.basename(excel_path),
        "source_file_full_path": os.path.abspath(excel_path),
        "output_folder": os.path.abspath(scenario_out_dir),
    }

    meta_path = os.path.join(scenario_out_dir, f"{prefix}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved files:")
    print(" ", os.path.abspath(r_path))
    print(" ", os.path.abspath(meta_path))

    print("\nSummary:")
    print(f"ATMs: {df[col_atm].nunique()}")
    print(f"Days: {len(unique_dates)}")
    print(f"Rows: {len(df)}")
    print(f"Folder: {os.path.abspath(scenario_out_dir)}")


# =========================
# USER INTERFACE
# =========================
def launch_ui():
    root = tk.Tk()
    root.title("ATM Optimization Pipeline")
    root.geometry("560x430")

    # -------- Scenario --------
    tk.Label(root, text="Scenario").pack(pady=(10, 5))

    scenario_var = tk.StringVar()
    scenario_box = ttk.Combobox(
        root,
        textvariable=scenario_var,
        values=["Scenario1", "Scenario2"],
        state="readonly",
        width=30
    )
    scenario_box.pack()
    scenario_box.current(0)

    # -------- Dates --------
    tk.Label(root, text="Planning Start Date (DD.MM.YYYY)").pack(pady=(15, 5))
    start_entry = tk.Entry(root, width=30)
    start_entry.pack()

    tk.Label(root, text="Planning End Date (DD.MM.YYYY)").pack(pady=(15, 5))
    end_entry = tk.Entry(root, width=30)
    end_entry.pack()

    tk.Label(root, text="Re-Optimization Date (DD.MM.YYYY)").pack(pady=(15, 5))
    reopt_entry = tk.Entry(root, width=30)
    reopt_entry.pack()

    # default values
    start_entry.insert(0, "26.02.2007")
    end_entry.insert(0, "04.03.2007")
    reopt_entry.insert(0, "25.02.2007")

    # -------- Reference file picker --------
    selected_file_var = tk.StringVar(value="No file selected")

    tk.Label(root, text="Reference Excel File").pack(pady=(15, 5))

    def browse_file():
        filepath = filedialog.askopenfilename(
            title="Select reference Excel file",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            selected_file_var.set(filepath)

    tk.Button(root, text="Choose Excel File", command=browse_file).pack()
    tk.Label(root, textvariable=selected_file_var, wraplength=520, justify="center").pack(pady=(8, 0))

    # -------- Run --------
    def run():
        scenario = scenario_var.get().strip()
        start_str = start_entry.get().strip()
        end_str = end_entry.get().strip()
        reopt_str = reopt_entry.get().strip()
        excel_path = selected_file_var.get().strip()

        if not excel_path or excel_path == "No file selected":
            messagebox.showerror("Error", "Please select a reference Excel file.")
            return

        if not os.path.isfile(excel_path):
            messagebox.showerror("Error", "Selected file does not exist.")
            return

        try:
            start_date = datetime.strptime(start_str, "%d.%m.%Y")
            end_date = datetime.strptime(end_str, "%d.%m.%Y")
            reopt_date = datetime.strptime(reopt_str, "%d.%m.%Y")
        except ValueError:
            messagebox.showerror("Error", "Please enter all dates in format DD.MM.YYYY")
            return

        if end_date < start_date:
            messagebox.showerror("Error", "Planning end date must be on or after planning start date.")
            return

        # prefix from scenario + planning dates
        start_tag = start_date.strftime("%Y%m%d")
        end_tag = end_date.strftime("%Y%m%d")
        prefix = f"{scenario}_{start_tag}_{end_tag}"

        try:
            build_r_json_from_excel(
                excel_path=excel_path,
                out_dir=OUT_DIR,
                prefix=prefix,
                planning_start=start_date,
                planning_end=end_date,
                reopt_date=reopt_date,
                col_date=COL_DATE,
                col_atm=COL_ATM,
                col_val=COL_VAL
            )
        except Exception as e:
            messagebox.showerror("Pipeline Error", str(e))
            return

        messagebox.showinfo(
            "Done",
            f"Pipeline finished successfully.\n\n"
            f"Prefix: {prefix}\n"
            f"Folder: {os.path.abspath(os.path.join(OUT_DIR, prefix))}"
        )

    tk.Button(root, text="Run Pipeline", command=run).pack(pady=20)

    root.mainloop()


# =========================
# START PROGRAM
# =========================
if __name__ == "__main__":
    launch_ui()