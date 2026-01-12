#!/usr/bin/env python3
"""
Generate LaTeX tables (2 per kernel = 6 total) from combined_results.csv.

- Table Kernel Chars (per kernel): known kernel properties + estimated kernel properties
- Table Stats (per kernel): performance metrics
    (AddedNoise, RecoveredNoise, CorrelationTime, SolveTime, RMSE, L2)

Notes:
- "Noise Level"    -> Added Noise (sigma_y) shown in the tables
- "FinalSigma"     -> Recovered Noise shown in the tables
- "SolveTime_min"  -> Solution time in minutes (added by deconvolution code)
- "SolveTime_sec"  -> Optional; not displayed unless you modify this script
- Truncated Linear method is skipped (not present in your CSV anyway)

Outputs:
  latex_tables/<noise_type>/
    gamma_<noise_type>_kernel_results.tex
    gamma_<noise_type>_performance_metrics.tex
    chapeau_<noise_type>_kernel_results.tex
    chapeau_<noise_type>_performance_metrics.tex
    bimodal_<noise_type>_kernel_results.tex
    bimodal_<noise_type>_performance_metrics.tex
"""

from __future__ import annotations

import os
import math
import pandas as pd
from typing import Dict, Tuple, Optional


# ---------------------------------------------
# Accumulate known kernel results into one csv:
# ---------------------------------------------
def combine_knwn_kernel_csvs(ws: str = ".", noise_type: str = "on-out"):
    curdir = os.getcwd()
    os.chdir(ws)
    folders = ["bimodal", "chapeau", "gamma"]
    rdf = pd.DataFrame()

    for folder in folders:
        subfolders = [
            f.path
            for f in os.scandir(os.path.join(folder, "outputs", noise_type))
            if f.is_dir()
        ]
        for subfolder in subfolders:
            noise_lvl = float(os.path.basename(subfolder).split("_")[-1])
            files = [f for f in os.listdir(subfolder) if f.endswith("_stats.csv")]
            for file in files:
                df = pd.read_csv(os.path.join(subfolder, file))
                df.insert(0, "Noise Level", noise_lvl)
                df.insert(0, "Known Kernel", folder)
                rdf = pd.concat([rdf, df], ignore_index=True)

    rdf = rdf.sort_values(by=["Known Kernel", "Noise Level", "Method"])
    os.chdir(curdir)
    rdf.to_csv(os.path.join(ws, "combined_results.csv"), index=False)


# -----------------------------
# FORMATTING HELPERS
# -----------------------------
def fmt_noise(x: float):
    """Format noise levels as in your examples (0.005 -> 0.005, 0.03 -> 0.03)."""
    if x < 0.01:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return f"{x:.2f}"


def fmt_val(x: float, ndp: int = 3):
    """Generic numeric formatting for table values."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.{ndp}f}"


def fmt_corr_time(x: float):
    """Correlation time formatting (keep 2 decimals; drop trailing zeros)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = f"{x:.2f}"
    return s.rstrip("0").rstrip(".")


def fmt_solve_time_min(x: float):
    """Solve time (minutes). Keep 2 decimals; drop trailing zeros."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    s = f"{x:.2f}"
    return s.rstrip("0").rstrip(".")


def latex_escape(s: str):
    """Minimal LaTeX escaping for text fields (kernel names)."""
    return (
        s.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


# -----------------------------
# TABLE GENERATORS
# -----------------------------
def get_known_props(df_k: pd.DataFrame, kernel: str):
    """
    Return (m0, m1, m2) known properties.
    If not provided in KNOWN_KERNEL_PROPS, estimate from minimum Noise Level rows.
    """
    k = kernel.lower()
    if k in {kk.lower(): kk for kk in KNOWN_KERNEL_PROPS}.keys():
        for kk, vv in KNOWN_KERNEL_PROPS.items():
            if kk.lower() == k:
                return vv

    min_noise = df_k["Noise Level"].min()
    df_min = df_k.loc[df_k["Noise Level"] == min_noise]
    m0 = float(df_min["Mass_m0"].mean())
    m1 = float(df_min["MeanTravelTime_m1"].mean())
    m2 = float(df_min["SpreadOfTravelTimes_m2"].mean())
    return (m0, m1, m2)


def kernel_results_table(df_k: pd.DataFrame, kernel: str):
    """
    LaTeX table: known kernel props + estimated props for each method and noise level.

    - Correlation Time column removed from this table.
    """
    kernel_clean = latex_escape(kernel)
    known_m0, known_m1, known_m2 = get_known_props(df_k, kernel)

    methods_present = [m for m in ["Cirpka", "Learn"] if m in df_k["Method"].unique()]

    lines = []
    lines.append("%" * 46)
    lines.append(r"\renewcommand{\arraystretch}{1.0}")
    lines.append(r"\begin{table}[H]")
    lines.append(
        rf"\caption{{Known {kernel_clean} kernel properties and estimated kernel properties with noise added to output signal ($ \y $)}}"
    )
    lines.append(r"\centering")
    lines.append("")
    lines.append(r"\begin{tabular}{")
    lines.append(r">{\centering\arraybackslash}p{2.0cm} ")
    lines.append(r">{\centering\arraybackslash}p{1.2cm} ")
    lines.append(r">{\centering\arraybackslash}p{1.2cm} ")
    lines.append(r">{\centering\arraybackslash}p{1.5cm} ")
    lines.append(r">{\centering\arraybackslash}p{1.5cm} ")
    lines.append(r"}")

    lines.append(
        r"\multicolumn{1}{c}{\small Method} & "
        r"\multicolumn{1}{c}{\small \shortstack{Added\\Noise ($\sigma_\y$)}} &"
        r"\multicolumn{1}{c}{\small \shortstack{Total Mass\\ ($m_0$)}} &"
        r"\multicolumn{1}{c}{\small \shortstack{Mean Travel\\Time ($m_1$)}} &"
        r"\multicolumn{1}{c}{\small \shortstack{Spread of\\Travel Times ($m_2$)}} \\"
    )

    lines.append("")
    lines.append(r"\toprule")

    lines.append(
        rf"\parbox[c]{{2.0cm}}{{\centering Known Kernel\\Properties:}} & -- & "
        rf"{fmt_val(known_m0,3)} & {fmt_val(known_m1,3)} & {fmt_val(known_m2,3)} \\"
    )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{c}{\textit{Estimated Kernels}} \\")
    lines.append(r"\midrule")

    for mi, method in enumerate(methods_present):
        df_m = df_k.loc[df_k["Method"] == method].copy()
        df_m = df_m.sort_values("Noise Level")
        label = METHOD_LABELS.get(method, latex_escape(method))

        nrows = len(df_m)
        if nrows == 0:
            continue

        first = True
        for _, r in df_m.iterrows():
            add_noise = fmt_noise(float(r["Noise Level"]))
            m0 = fmt_val(float(r["Mass_m0"]), 3)
            m1 = fmt_val(float(r["MeanTravelTime_m1"]), 3)
            m2 = fmt_val(float(r["SpreadOfTravelTimes_m2"]), 3)

            if first:
                lines.append(
                    rf"\multirow{{{nrows}}}{{*}}{{{label}}} & {add_noise} & {m0} & {m1} & {m2} \\"
                )
                first = False
            else:
                lines.append(
                    rf"                        & {add_noise} & {m0} & {m1} & {m2} \\"
                )

        if mi < len(methods_present) - 1:
            lines.append(r"\cmidrule(lr){2-5}")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"\label{{tab:{kernel.lower()}_{noise_type}_kernel_results}}")
    lines.append(r"\end{table}")
    lines.append("%" * 46)
    lines.append("")
    return "\n".join(lines)


def performance_metrics_table(df_k: pd.DataFrame, kernel: str):
    """
    LaTeX table: metrics per method and noise level.

    Columns:
      Method | AddedNoise | RecoveredNoise | CorrelationTime | SolveTime | RMSE | L2
    """
    kernel_clean = latex_escape(kernel)
    methods_present = [m for m in ["Cirpka", "Learn"] if m in df_k["Method"].unique()]

    lines = []
    lines.append("%" * 46)
    lines.append(r"\renewcommand{\arraystretch}{1.0}")
    lines.append(r"\begin{table}[H]")
    lines.append(
        rf"\caption{{Performance metrics using the {kernel_clean} kernel with noise added to output signal ($ \y $)}}"
    )
    lines.append(r"\centering")
    lines.append("")
    lines.append(r"\begin{tabular}{")
    lines.append(r">{\centering\arraybackslash}p{2.4cm} ")  # Method
    lines.append(r">{\centering\arraybackslash}p{1.3cm} ")  # Added noise
    lines.append(r">{\centering\arraybackslash}p{1.8cm} ")  # Recovered noise
    lines.append(r">{\centering\arraybackslash}p{1.5cm} ")  # Corr time
    lines.append(r">{\centering\arraybackslash}p{1.5cm} ")  # Solve time
    lines.append(r">{\centering\arraybackslash}p{1.6cm} ")  # RMSE
    lines.append(r">{\centering\arraybackslash}p{1.4cm} ")  # L2
    lines.append(r"}")

    lines.append(
        r"\small Method &"
        r"\multicolumn{1}{c}{\small \shortstack{Added\\Noise ($\sigma_\y$)}} &"
        r"\multicolumn{1}{c}{\small \shortstack{Recovered\\Noise}} &"
        r"\multicolumn{1}{c}{\small \shortstack{Correlation\\Time (hr)}} &"
        r"\multicolumn{1}{c}{\small \shortstack{Solve\\Time (min)}} &"
        r"\multicolumn{1}{c}{\small \shortstack{RMSE\\($\X\bar\g$, $\y$)}} &"
        r"\small $L^2$ ($\bar\g,\k$) \\"
    )
    lines.append("")
    lines.append(r"\hline")

    for method in methods_present:
        df_m = df_k.loc[df_k["Method"] == method].copy()
        df_m = df_m.sort_values("Noise Level")
        label = METHOD_LABELS.get(method, latex_escape(method))

        nrows = len(df_m)
        if nrows == 0:
            continue

        first = True
        for _, r in df_m.iterrows():
            add_noise = fmt_noise(float(r["Noise Level"]))
            rec_noise = fmt_val(float(r["FinalSigma"]), 3)
            ct = fmt_corr_time(float(r["CorrelationTime"]))
            st = fmt_solve_time_min(float(r["SolveTime_min"])) if "SolveTime_min" in df_m.columns else ""
            rmse = fmt_val(float(r["RMSE"]), 4)
            l2 = fmt_val(float(r["L2"]), 4)

            if first:
                lines.append(rf"\multirow{{{nrows}}}{{*}}{{{label}}}")
                lines.append(
                    rf"    & {add_noise} & {rec_noise} & {ct} & {st} & {rmse} & {l2} \\"
                )
                first = False
            else:
                lines.append(
                    rf"    & {add_noise} & {rec_noise} & {ct} & {st} & {rmse} & {l2} \\"
                )

        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(rf"\label{{tab:{kernel.lower()}_{noise_type}_performance_metrics}}")
    lines.append(r"\end{table}")
    lines.append("%" * 46)
    lines.append("")
    return "\n".join(lines)


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    required = [
        "Known Kernel",
        "Noise Level",
        "Method",
        "Mass_m0",
        "MeanTravelTime_m1",
        "SpreadOfTravelTimes_m2",
        "RMSE",
        "CorrelationTime",
        "FinalSigma",
        "L2",
        # SolveTime_min is optional but strongly recommended; we check below.
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # numeric columns
    num_cols = [
        "Noise Level",
        "Mass_m0",
        "MeanTravelTime_m1",
        "SpreadOfTravelTimes_m2",
        "RMSE",
        "CorrelationTime",
        "FinalSigma",
        "L2",
    ]
    if "SolveTime_min" in df.columns:
        num_cols.append("SolveTime_min")

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["Method"].isin(METHOD_LABELS.keys())].copy()

    kernels = KERNELS if KERNELS is not None else sorted(df["Known Kernel"].unique())

    for kernel in kernels:
        df_k = df[df["Known Kernel"] == kernel].copy()
        if df_k.empty:
            continue

        tex1 = kernel_results_table(df_k, kernel)
        tex2 = performance_metrics_table(df_k, kernel)

        out1 = os.path.join(OUTDIR, f"{kernel.lower()}_{noise_type}_kernel_results.tex")
        out2 = os.path.join(OUTDIR, f"{kernel.lower()}_{noise_type}_performance_metrics.tex")

        with open(out1, "w", encoding="utf-8") as f:
            f.write(tex1)
        with open(out2, "w", encoding="utf-8") as f:
            f.write(tex2)

        print(f"Wrote: {out1}")
        print(f"Wrote: {out2}")


if __name__ == "__main__":
    # workspace:
    ws = os.path.join("known_kernels", "python_make")
    # noise type:
    noise_type = 'on-in-after-conv'
    # combine results csvs:
    combine_knwn_kernel_csvs(ws=ws, noise_type=noise_type)

    CSV_PATH = os.path.join(ws, "combined_results.csv")
    OUTDIR = os.path.join(ws, "latex_tables", noise_type)

    # ***HARD coded known kernel props:
    KNOWN_KERNEL_PROPS: Dict[str, Tuple[float, float, float]] = {
        "gamma": (1.0, 6.0, 12.0),
        "chapeau": (1.0, 2.5, 1.2),
        "bimodal": (1.0, 12.63, 22.95),
    }

    # how to display method names in LaTeX
    METHOD_LABELS = {
        "Cirpka": r"\shortstack{Modified\\Cirpka}",
        "Learn": r"Learn",
    }

    # kernels to generate tables for (if None, infer from CSV unique values)
    KERNELS: Optional[list[str]] = None

    # generate latex tables:
    main()
