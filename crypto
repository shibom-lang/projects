#!/usr/bin/env python3
"""Enhanced distribution diagnostics for high-frequency return series."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def compute_metrics(x: np.ndarray) -> dict[str, float]:
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    skew = stats.skew(x, bias=False)
    ex_kurt = stats.kurtosis(x, fisher=True, bias=False)
    jb_stat, jb_p = stats.jarque_bera(x)
    sw_stat, sw_p = stats.shapiro(x[: min(len(x), 5000)]) if len(x) > 3 else (np.nan, np.nan)
    return {
        "n": len(x),
        "mean": mu,
        "std": sigma,
        "skew": skew,
        "ex_kurt": ex_kurt,
        "jb_stat": jb_stat,
        "jb_p": jb_p,
        "sw_stat": sw_stat,
        "sw_p": sw_p,
    }


def empirical_var_es(x: np.ndarray, alpha: float) -> tuple[float, float]:
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    return q, (np.mean(tail) if tail.size else np.nan)


def normal_var_es(mu: float, sigma: float, alpha: float) -> tuple[float, float]:
    z = stats.norm.ppf(alpha)
    return mu + sigma * z, mu - sigma * stats.norm.pdf(z) / alpha


def t_var_es(df: float, loc: float, scale: float, alpha: float) -> tuple[float, float]:
    q = stats.t.ppf(alpha, df=df, loc=loc, scale=scale)
    x_alpha = stats.t.ppf(alpha, df=df)
    pdf = stats.t.pdf(x_alpha, df=df)
    es_std = -((df + x_alpha**2) / (df - 1)) * (pdf / alpha) if df > 1 else np.nan
    return q, loc + scale * es_std


def rolling_shape(series: pd.Series, window: int) -> pd.DataFrame:
    return pd.DataFrame({"rolling_skew": series.rolling(window).skew(), "rolling_ex_kurt": series.rolling(window).kurt()})


def plot_dashboard(df: pd.DataFrame, col: str, title: str, out: Path, window: int, tail_q: float) -> None:
    x = df[col].dropna().to_numpy()
    if x.size < 50:
        raise ValueError("Need at least 50 observations for stable diagnostics.")

    m = compute_metrics(x)
    mu, sigma = m["mean"], m["std"]
    t_df, t_loc, t_scale = stats.t.fit(x)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    bins = min(250, max(60, int(np.sqrt(len(x)))))
    ax1.hist(x, bins=bins, density=True, alpha=0.55, color="#8b5cf6", edgecolor="#4c1d95", linewidth=0.4, label="Empirical")
    grid = np.linspace(np.quantile(x, 1 - tail_q), np.quantile(x, tail_q), 1800)
    ax1.plot(grid, stats.norm.pdf(grid, mu, sigma), color="#dc2626", lw=2.2, label="Normal fit")
    ax1.plot(grid, stats.t.pdf(grid, t_df, t_loc, t_scale), color="#0ea5e9", lw=2.2, label=f"Student-t fit (ν={t_df:.2f})")
    ax1.set_title("Distribution Overlay")
    ax1.set_xlabel("Log Return")
    ax1.set_ylabel("Density")
    ax1.legend(frameon=False)

    stats.probplot(x, dist="norm", plot=ax2)
    ax2.set_title("Normal Q-Q Plot")

    abs_x = np.sort(np.abs(x))
    surv = 1.0 - np.arange(1, len(abs_x) + 1) / len(abs_x)
    mask = surv > 0
    ax3.plot(abs_x[mask], surv[mask], color="#7c3aed", lw=1.3)
    ax3.set_yscale("log")
    ax3.set_title("Tail Survival (log scale)")
    ax3.set_xlabel("|Return|")
    ax3.set_ylabel("P(|R| > x)")

    roll = rolling_shape(df[col].dropna(), window)
    ax4.plot(roll.index, roll["rolling_skew"], label="Rolling skew", color="#0ea5e9")
    ax4.axhline(0, color="gray", ls="--", lw=1)
    ax4.set_title(f"Rolling Skew ({window} obs)")
    ax4.legend(frameon=False)

    ax5.plot(roll.index, roll["rolling_ex_kurt"], label="Rolling excess kurtosis", color="#f59e0b")
    ax5.axhline(0, color="gray", ls="--", lw=1)
    ax5.set_title(f"Rolling Excess Kurtosis ({window} obs)")
    ax5.legend(frameon=False)

    rows = []
    for alpha in (0.05, 0.01):
        e_var, e_es = empirical_var_es(x, alpha)
        n_var, n_es = normal_var_es(mu, sigma, alpha)
        tt_var, tt_es = t_var_es(t_df, t_loc, t_scale, alpha)
        rows.append((f"{int(alpha * 100)}%", e_var, n_var, tt_var, e_es, n_es, tt_es, n_var - e_var))

    risk = pd.DataFrame(rows, columns=["Tail", "Emp VaR", "Norm VaR", "t VaR", "Emp ES", "Norm ES", "t ES", "Norm VaR Bias"])
    ax6.axis("off")
    tbl = ax6.table(cellText=np.round(risk.values, 7), colLabels=risk.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)
    ax6.set_title("Tail Risk Comparison")

    fig.suptitle(f"{title}\nSkew={m['skew']:+.4f} | Excess Kurtosis={m['ex_kurt']:.2f} | JB p-value={m['jb_p']:.2e}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=180)
    plt.close(fig)

    left_1pct = np.quantile(x, 0.01)
    right_99pct = np.quantile(x, 0.99)
    asymmetry_ratio = abs(right_99pct / left_1pct) if left_1pct != 0 else np.nan

    out.with_suffix(".summary.txt").write_text(
        "\n".join(
            [
                f"samples={m['n']}",
                f"mean={m['mean']:.8f}",
                f"std={m['std']:.8f}",
                f"skew={m['skew']:.6f}",
                f"excess_kurtosis={m['ex_kurt']:.6f}",
                f"jarque_bera_stat={m['jb_stat']:.6f}",
                f"jarque_bera_p={m['jb_p']:.6e}",
                f"student_t_df={t_df:.6f}",
                f"student_t_loc={t_loc:.8f}",
                f"student_t_scale={t_scale:.8f}",
                f"left_1pct_quantile={left_1pct:.8f}",
                f"right_99pct_quantile={right_99pct:.8f}",
                f"tail_asymmetry_ratio_abs(q99/q01)={asymmetry_ratio:.6f}",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate return distribution diagnostics.")
    parser.add_argument("--csv", required=True, help="Path to CSV containing return data")
    parser.add_argument("--col", required=True, help="Column with return values")
    parser.add_argument("--out", default="return_diagnostics.png", help="Output figure path")
    parser.add_argument("--title", default="1-Minute Log Returns Diagnostic")
    parser.add_argument("--window", type=int, default=1440, help="Rolling window length in observations")
    parser.add_argument("--tail-q", type=float, default=0.999, help="Upper quantile cutoff for overlay grid (default: 0.999)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.col not in df.columns:
        raise ValueError(f"Column '{args.col}' not found. Available columns: {list(df.columns)}")
    if not (0.9 <= args.tail_q < 1.0):
        raise ValueError("--tail-q should be in [0.9, 1.0).")

    plot_dashboard(df, args.col, args.title, Path(args.out), args.window, args.tail_q)
    print(f"Saved diagnostics to {args.out}")


if __name__ == "__main__":
    main()
