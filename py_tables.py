#!/usr/bin/env python3
"""
Generate LaTeX tables (2 per kernel = 6 total) from combined_results.csv.

- Table Kernel Chars (per kernel): known kernel properties + estimated kernel properties
    Columns:
      Method | Added Noise | Correlation Time | Total Mass | Mean Travel Time | Spread of Travel Times
- Table Stats (per kernel): performance metrics
    Columns:
      Method | Added Noise | Recovered Noise | Solve Time | RMSE | L2

Noise-label + caption logic:
- noise_type == 'on-out'
    Added noise: sigma_y added to output signal (y)
- noise_type == 'on-in-before-conv'
    Added noise: sigma_{x^{*}} added to input signal (x*) before convolution
- noise_type == 'on-in-after-conv'
    Added noise: sigma_x added to the input signal only after generating the output
                signal via convolution (i.e., noise added post-convolution)

Notes:
- "Noise Level"    -> Added Noise shown in the tables (sigma symbol depends on noise_type)
- "FinalSigma"     -> Recovered Noise shown in the tables
- "SolveTime_min"  -> Solution time in minutes (added by deconvolution code)
"""

from __future__ import annotations

import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, Tuple, Optional


# ---------------------------------------------
# Accum known kernel results into one csv:
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
#  Formats
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
    """Correlation time formatting (keep 3 decimals; drop trailing zeros)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    s = f"{x:.3f}"
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
# Noise types
# -----------------------------
def noise_symbol(noise_type: str) -> str:
    """
    Return the LaTeX symbol for "Added Noise" for this noise_type.

    - on-out            -> sigma_y
    - on-in-before-conv -> sigma_{x^{*}}  (xstar)
    - on-in-after-conv  -> sigma_x
    """
    if noise_type == "on-in-before-conv":
        return r"\sigma_{\x^{\!*}}"
    if noise_type == "on-in-after-conv":
        return r"\sigma_{\x}"
    return r"\sigma_{\y}"


def added_noise_header(noise_type: str) -> str:
    sym = noise_symbol(noise_type)
    return rf"\multicolumn{{1}}{{c}}{{\small \shortstack{{Added\\Noise (${sym}$)}}}} &"


def noise_caption_phrase(noise_type: str) -> str:
    """
    Caption phrase describing where noise is added.
    """
    if noise_type == "on-in-before-conv":
        return rf"noise added to the input signal ($ {noise_symbol(noise_type)} $) before convolution"
    if noise_type == "on-in-after-conv":
        return rf"noise added to the input signal ($ {noise_symbol(noise_type)} $) only after generating the output signal with a convolution"
    return rf"noise added to the output signal ($ {noise_symbol(noise_type)} $)"


# -----------------------------
# Table gens
# -----------------------------
def get_known_props(df_k: pd.DataFrame, kernel: str):
    """
    Return (corr_time, m0, m1, m2) known properties.
    If m0, m1, m2 are not provided in KNOWN_KERNEL_PROPS, estimate from minimum Noise Level rows.
    """
    k = kernel.lower()

    known_corr = None
    for kk, vv in KNOWN_CORR_TIMES.items():
        if kk.lower() == k:
            known_corr = vv
            break

    known_m = None
    for kk, vv in KNOWN_KERNEL_PROPS.items():
        if kk.lower() == k:
            known_m = vv
            break

    if known_m is not None:
        known_m0, known_m1, known_m2 = known_m
    else:
        min_noise = df_k["Noise Level"].min()
        df_min = df_k.loc[df_k["Noise Level"] == min_noise]
        known_m0 = float(df_min["Mass_m0"].mean())
        known_m1 = float(df_min["MeanTravelTime_m1"].mean())
        known_m2 = float(df_min["SpreadOfTravelTimes_m2"].mean())

    return known_corr, known_m0, known_m1, known_m2


def kernel_results_table(df_k: pd.DataFrame, kernel: str, noise_type: str):
    """
    LaTeX table: known kernel props + estimated props for each method and noise level.

    Columns:
      Method | Added Noise | Correlation Time | Total Mass | Mean Travel Time | Spread of Travel Times
    """
    kernel_clean = latex_escape(kernel)
    known_ct, known_m0, known_m1, known_m2 = get_known_props(df_k, kernel)

    methods_present = [m for m in ["Cirpka", "Learn"] if m in df_k["Method"].unique()]

    add_hdr = added_noise_header(noise_type)

    lines = []
    lines.append("%" * 46)
    lines.append(r"\renewcommand{\arraystretch}{1.0}")
    lines.append(r"\begin{table}[H]")
    lines.append(
        rf"\caption{{Known {kernel_clean} kernel properties and estimated kernel properties with {noise_caption_phrase(noise_type)}.}}"
    )
    lines.append(r"\centering")
    lines.append("")
    lines.append(r"\begin{tabular}{")
    lines.append(r">{\centering\arraybackslash}p{2.0cm} ")
    lines.append(r">{\centering\arraybackslash}p{1.2cm} ")
    lines.append(r">{\centering\arraybackslash}p{1.5cm} ")
    lines.append(r">{\centering\arraybackslash}p{1.2cm} ")
    lines.append(r">{\centering\arraybackslash}p{1.5cm} ")
    lines.append(r">{\centering\arraybackslash}p{1.6cm} ")
    lines.append(r"}")

    lines.append(
        r"\multicolumn{1}{c}{\small Method} & "
        + add_hdr
        + r" \multicolumn{1}{c}{\small \shortstack{Correlation\\Time (hr)}} & "
        + r"\multicolumn{1}{c}{\small \shortstack{Total Mass\\($m_0$)}} & "
        + r"\multicolumn{1}{c}{\small \shortstack{Mean Travel\\Time ($m_1$)}} & "
        + r"\multicolumn{1}{c}{\small \shortstack{Spread of\\Travel Times ($m_2$)}} \\"
    )

    lines.append("")
    lines.append(r"\toprule")

    lines.append(
        rf"\parbox[c]{{2.0cm}}{{\centering Known Kernel\\Properties:}} & -- & "
        rf"{fmt_corr_time(known_ct)} & {fmt_val(known_m0,3)} & {fmt_val(known_m1,3)} & {fmt_val(known_m2,3)} \\"
    )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{c}{\textit{Estimated Kernels}} \\")
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
            ct = fmt_corr_time(float(r["CorrelationTime"])/2)
            m0 = fmt_val(float(r["Mass_m0"]), 3)
            m1 = fmt_val(float(r["MeanTravelTime_m1"]), 3)
            m2 = fmt_val(float(r["SpreadOfTravelTimes_m2"]), 3)

            if first:
                lines.append(
                    rf"\multirow{{{nrows}}}{{*}}{{{label}}} & {add_noise} & {ct} & {m0} & {m1} & {m2} \\"
                )
                first = False
            else:
                lines.append(
                    rf"                        & {add_noise} & {ct} & {m0} & {m1} & {m2} \\"
                )

        if mi < len(methods_present) - 1:
            lines.append(r"\cmidrule(lr){2-6}")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"\label{{tab:{kernel.lower()}_{noise_type}_kernel_results}}")
    lines.append(r"\end{table}")
    lines.append("%" * 46)
    lines.append("")
    return "\n".join(lines)


def performance_metrics_table(df_k: pd.DataFrame, kernel: str, noise_type: str):
    """
    LaTeX table: metrics per method and noise level.

    Columns:
      Method | AddedNoise | RecoveredNoise | SolveTime | RMSE | L2
    """
    kernel_clean = latex_escape(kernel)
    methods_present = [m for m in ["Cirpka", "Learn"] if m in df_k["Method"].unique()]

    add_hdr = added_noise_header(noise_type)

    lines = []
    lines.append("%" * 46)
    lines.append(r"\renewcommand{\arraystretch}{1.0}")
    lines.append(r"\begin{table}[H]")
    lines.append(
        rf"\caption{{Performance metrics using the {kernel_clean} kernel with {noise_caption_phrase(noise_type)}.}}"
    )
    lines.append(r"\centering")
    lines.append("")
    lines.append(r"\begin{tabular}{")
    lines.append(r">{\centering\arraybackslash}p{2.4cm} ")  # Method
    lines.append(r">{\centering\arraybackslash}p{1.3cm} ")  # Added noise
    lines.append(r">{\centering\arraybackslash}p{1.8cm} ")  # Recovered noise
    lines.append(r">{\centering\arraybackslash}p{1.5cm} ")  # Solve time
    lines.append(r">{\centering\arraybackslash}p{1.6cm} ")  # RMSE
    lines.append(r">{\centering\arraybackslash}p{1.4cm} ")  # L2
    lines.append(r"}")

    lines.append(
        r"\small Method &"
        + add_hdr
        + r" \multicolumn{1}{c}{\small \shortstack{Recovered\\Noise}} &"
        + r"\multicolumn{1}{c}{\small \shortstack{Solve\\Time (min)}} &"
        + r"\multicolumn{1}{c}{\small \shortstack{RMSE\\($\X\bar\g$, $\y$)}} &"
        + r"\small $L^2$ ($\bar\g,\k$) \\"
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
            st = (
                fmt_solve_time_min(float(r["SolveTime_min"]))
                if "SolveTime_min" in df_m.columns
                else ""
            )
            rmse = fmt_val(float(r["RMSE"]), 4)
            l2 = fmt_val(float(r["L2"]), 4)

            if first:
                lines.append(rf"\multirow{{{nrows}}}{{*}}{{{label}}}")
                lines.append(
                    rf"    & {add_noise} & {rec_noise} & {st} & {rmse} & {l2} \\"
                )
                first = False
            else:
                lines.append(
                    rf"    & {add_noise} & {rec_noise} & {st} & {rmse} & {l2} \\"
                )

        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(rf"\label{{tab:{kernel.lower()}_{noise_type}_performance_metrics}}")
    lines.append(r"\end{table}")
    lines.append("%" * 46)
    lines.append("")
    return "\n".join(lines)


# -----------------------------
# PDF RENDER (same content as the .tex tables, no LaTeX toolchain needed)
# -----------------------------
def noise_symbol_mathtext(noise_type: str) -> str:
    """Mathtext version of noise_symbol() for matplotlib rendering."""
    if noise_type == "on-in-before-conv":
        return r"$\sigma_{x^*}$"
    if noise_type == "on-in-after-conv":
        return r"$\sigma_x$"
    return r"$\sigma_y$"


def noise_caption_plain(noise_type: str) -> str:
    """Plain-text caption phrase (mirrors noise_caption_phrase)."""
    sym = noise_symbol_mathtext(noise_type)
    if noise_type == "on-in-before-conv":
        return f"noise added to the input signal ({sym}) before convolution"
    if noise_type == "on-in-after-conv":
        return (f"noise added to the input signal ({sym}) only after generating "
                "the output signal with a convolution")
    return f"noise added to the output signal ({sym})"


def _method_label_plain(method: str) -> str:
    return {"Cirpka": "Modified Cirpka", "Learn": "Learn"}.get(method, method)


def kernel_results_rows(df_k: pd.DataFrame, kernel: str):
    """Row data for the kernel-properties table (mirrors kernel_results_table)."""
    known_ct, known_m0, known_m1, known_m2 = get_known_props(df_k, kernel)
    rows = [["Known Kernel Properties", "--", fmt_corr_time(known_ct),
             fmt_val(known_m0, 3), fmt_val(known_m1, 3), fmt_val(known_m2, 3)]]
    for method in [m for m in ["Cirpka", "Learn"] if m in df_k["Method"].unique()]:
        df_m = df_k.loc[df_k["Method"] == method].sort_values("Noise Level")
        for i, (_, r) in enumerate(df_m.iterrows()):
            rows.append([
                _method_label_plain(method) if i == 0 else "",
                fmt_noise(float(r["Noise Level"])),
                fmt_corr_time(float(r["CorrelationTime"]) / 2),
                fmt_val(float(r["Mass_m0"]), 3),
                fmt_val(float(r["MeanTravelTime_m1"]), 3),
                fmt_val(float(r["SpreadOfTravelTimes_m2"]), 3),
            ])
    return rows


def performance_rows(df_k: pd.DataFrame):
    """Row data for the performance table (mirrors performance_metrics_table)."""
    rows = []
    for method in [m for m in ["Cirpka", "Learn"] if m in df_k["Method"].unique()]:
        df_m = df_k.loc[df_k["Method"] == method].sort_values("Noise Level")
        for i, (_, r) in enumerate(df_m.iterrows()):
            st = fmt_solve_time_min(float(r["SolveTime_min"])) \
                if "SolveTime_min" in df_m.columns else ""
            rows.append([
                _method_label_plain(method) if i == 0 else "",
                fmt_noise(float(r["Noise Level"])),
                fmt_val(float(r["FinalSigma"]), 3),
                st,
                fmt_val(float(r["RMSE"]), 4),
                fmt_val(float(r["L2"]), 4),
            ])
    return rows


def _draw_table(ax, col_labels, rows, title, hline_after=None):
    """Render one table onto an axis, styled to read like the LaTeX version."""
    ax.axis("off")
    ax.set_title(title, fontsize=9, loc="left", pad=14, wrap=True)
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="upper center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.0)
        if r == 0:                                  # header
            cell.set_text_props(weight="bold")
            cell.visible_edges = "B"
            cell.set_linewidth(1.0)
        elif hline_after and r in hline_after:      # rule under a group
            cell.visible_edges = "B"
            cell.set_linewidth(0.6)
        if r == len(rows):                          # bottom rule
            cell.visible_edges = ("B" if cell.visible_edges == "" else "B")
            cell.set_linewidth(1.0)
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
    return tbl


def write_tables_pdf(df, kernels, noise_type: str, outpath: str):
    """Write ALL tables for one noise type into a single multi-page PDF
    (one page per kernel: kernel properties on top, performance below)."""
    sym = noise_symbol_mathtext(noise_type)
    cols_kr = ["Method", f"Added\nNoise ({sym})", "Correlation\nTime (hr)",
               "Total Mass\n($m_0$)", "Mean Travel\nTime ($m_1$)",
               "Spread of\nTravel Times ($m_2$)"]
    cols_pm = ["Method", f"Added\nNoise ({sym})", "Recovered\nNoise",
               "Solve\nTime (min)", "RMSE\n($Xg$, $y$)", "$L^2$ ($g$, $k$)"]

    with PdfPages(outpath) as pdf:
        for kernel in kernels:
            df_k = df[df["Known Kernel"] == kernel].copy()
            if df_k.empty:
                continue
            rows_kr = kernel_results_rows(df_k, kernel)
            rows_pm = performance_rows(df_k)

            # size the page to the content so the tables aren't lost in whitespace
            n1, n2 = len(rows_kr) + 1, len(rows_pm) + 1
            fig_h = 1.6 + 0.30 * (n1 + n2)
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8.5, fig_h),
                gridspec_kw={'height_ratios': [n1, n2], 'hspace': 0.5})
            fig.suptitle(f"{kernel.capitalize()} kernel — {noise_type}",
                         fontsize=12, fontweight="bold")
            _draw_table(ax1, cols_kr, rows_kr,
                        f"Known {kernel} kernel properties and estimated kernel "
                        f"properties with {noise_caption_plain(noise_type)}.",
                        hline_after={1})
            _draw_table(ax2, cols_pm, rows_pm,
                        f"Performance metrics using the {kernel} kernel with "
                        f"{noise_caption_plain(noise_type)}.")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Wrote: {outpath}")



def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

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
     
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

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

        tex1 = kernel_results_table(df_k, kernel, noise_type=noise_type)
        tex2 = performance_metrics_table(df_k, kernel, noise_type=noise_type)

        out1 = os.path.join(OUTDIR, f"{kernel.lower()}_{noise_type}_kernel_results.tex")
        out2 = os.path.join(
            OUTDIR, f"{kernel.lower()}_{noise_type}_performance_metrics.tex"
        )

        with open(out1, "w", encoding="utf-8") as f:
            f.write(tex1)
        with open(out2, "w", encoding="utf-8") as f:
            f.write(tex2)

        print(f"Wrote: {out1}")
        print(f"Wrote: {out2}")

    # all tables for this noise type in one PDF
    write_tables_pdf(df, kernels, noise_type,
                     os.path.join(OUTDIR, f"tables_{noise_type}.pdf"))


if __name__ == "__main__":
    ws = os.path.join("known_kernels", "python_make")

    # noise types:
    noise_types = ["on-out", "on-in-before-conv", "on-in-after-conv"]
    
    for noise_type in noise_types:
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

        # ***HARD coded known correlation times (hr):
        KNOWN_CORR_TIMES: Dict[str, float] = {
            "chapeau": 1.924,
            "gamma": 5.383,
            "bimodal": 7.68,
        }

        METHOD_LABELS = {
            "Cirpka": r"\shortstack{Modified\\Cirpka}",
            "Learn": r"Learn",
        }

        KERNELS: Optional[list[str]] = None

        # gen latex tables:
        main()