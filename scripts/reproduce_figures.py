"""
Reproduce the paper's policy-comparison figures:

1. For each baseline policy individually, (T_baseline - T_opt) / T_baseline
   as a heatmap over the (p, f0) plane (one panel per baseline: pumping,
   nested, greedy).
2. The properly-normalized joint comparison, (T_min(baselines) - T_opt) /
   T_min(baselines), as a heatmap over the same (p, f0) plane.
3. T_opt vs. m (number of memories per node), for several values of p, at
   fixed f0 and delta.

Usage:
    python scripts/reproduce_figures.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.mdp import build_mdp
from src.optimal_policy import value_iteration
from src.baselines import build_pumping_policy, build_nested_policy, build_greedy_policy
from src.policy_eval import expected_waiting_time

BASELINE_BUILDERS = {
    "pumping": build_pumping_policy,
    "nested": build_nested_policy,
    "greedy": build_greedy_policy,
}


# =========================================================
# -----(p, f0) SWEEP: OPTIMAL VS. EACH BASELINE-----
# =========================================================
def sweep_pf0(p_vals, f0_vals, m, delta):
    """Sweeps over (p, f0) for fixed m and fidelity gap delta (fT = f0 +
    delta). Evaluates the optimal policy and all three baselines at each
    point."""
    results = []
    for p in p_vals:
        for f0 in f0_vals:
            mdp = build_mdp(p, m, f0, delta)
            V, _ = value_iteration(mdp)
            init_id = mdp["state2id"][tuple([0] * m)]
            T_opt = -V[init_id]

            baselines = {name: expected_waiting_time(mdp, builder(mdp))
                         for name, builder in BASELINE_BUILDERS.items()}
            best_name = min(baselines, key=baselines.get)
            T_min_base = baselines[best_name]
            adv_min = (T_min_base - T_opt) / T_min_base if np.isfinite(T_min_base) else np.nan

            row = dict(p=p, f0=f0, K=mdp["F"], T_opt=T_opt, best_baseline=best_name,
                       T_min_base=T_min_base, advantage_min=adv_min)
            for name, T in baselines.items():
                row[f"T_{name}"] = T
                row[f"advantage_{name}"] = (T - T_opt) / T if np.isfinite(T) else np.nan
            results.append(row)
            print(f"p={p:.2f} f0={f0:.3f} K={mdp['F']:2d} T_opt={T_opt:7.3f} "
                  f"best={best_name:7s} adv_min={adv_min*100:5.1f}%", flush=True)
    return results


# =========================================================
# -----PLOTTING: ONE HEATMAP PER BASELINE-----
# =========================================================
def plot_per_baseline_heatmaps(results, p_vals, f0_vals, m, delta,
                                outfile="figures/advantage_per_baseline.png"):
    """(T_baseline - T_opt) / T_baseline over the (p, f0) plane, one panel
    per baseline (pumping, nested, greedy)."""
    names = list(BASELINE_BUILDERS.keys())
    fig, axes = plt.subplots(1, len(names),
                              figsize=(1.5 * len(f0_vals) * len(names) / 1.3,
                                       1.1 * len(p_vals) + 1.5),
                              sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        grid = np.full((len(p_vals), len(f0_vals)), np.nan)
        for r in results:
            i, j = p_vals.index(r["p"]), f0_vals.index(r["f0"])
            grid[i, j] = r[f"advantage_{name}"] * 100
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                        vmin=0, vmax=max(1, np.nanmax(grid)))
        ax.set_xticks(range(len(f0_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in f0_vals], rotation=45)
        ax.set_yticks(range(len(p_vals)))
        ax.set_yticklabels([f"{v:.2f}" for v in p_vals])
        ax.set_xlabel(r"$f_0$")
        ax.set_title(f"vs. {name}")
        for i in range(len(p_vals)):
            for j in range(len(f0_vals)):
                val = grid[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val < 0.5 * np.nanmax(grid) else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                         fontsize=7, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel(r"$p$")
    fig.suptitle(
        r"$(T_{\mathrm{baseline}}-T_{\mathrm{opt}})/T_{\mathrm{baseline}}$"
        f"   [m={m}, " + r"$\Delta f$" + f"={delta}]"
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


# =========================================================
# -----PLOTTING: MIN-OVER-BASELINES HEATMAP-----
# =========================================================
def plot_advantage_min(results, p_vals, f0_vals, m, delta,
                        outfile="figures/advantage_heatmap.png"):
    """(T_min(baselines) - T_opt) / T_min(baselines) over the (p, f0) plane."""
    short = {"pumping": "P", "nested": "N", "greedy": "G"}
    grid = np.full((len(p_vals), len(f0_vals)), np.nan)
    labels = [["" for _ in f0_vals] for _ in p_vals]
    for r in results:
        i, j = p_vals.index(r["p"]), f0_vals.index(r["f0"])
        grid[i, j] = r["advantage_min"] * 100
        labels[i][j] = f"{r['advantage_min']*100:.0f}%\n({short[r['best_baseline']]})"

    fig, ax = plt.subplots(figsize=(1.6 * len(f0_vals), 1.1 * len(p_vals) + 1.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(f0_vals))); ax.set_xticklabels([f"{v:.2f}" for v in f0_vals])
    ax.set_yticks(range(len(p_vals))); ax.set_yticklabels([f"{v:.2f}" for v in p_vals])
    ax.set_xlabel(r"$f_0$"); ax.set_ylabel(r"$p$")
    ax.set_title(
        r"$(T_{\min(\mathrm{baselines})}-T_{\mathrm{opt}})/T_{\min(\mathrm{baselines})}$"
        f"   [m={m}, " + r"$\Delta f$" + f"={delta}]"
    )
    for i in range(len(p_vals)):
        for j in range(len(f0_vals)):
            val = grid[i, j]
            color = "white" if (not np.isnan(val) and val < 20) else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax).set_label("% reduction vs. best baseline")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


# =========================================================
# -----(m) SWEEP: T_opt VS. m FOR SEVERAL p-----
# =========================================================
def sweep_m(p_vals, m_vals, f0, delta):
    """T_opt vs. m for several values of p, at fixed f0 and delta."""
    results = []
    for p in p_vals:
        for m in m_vals:
            mdp = build_mdp(p, m, f0, delta)
            V, _ = value_iteration(mdp)
            init_id = mdp["state2id"][tuple([0] * m)]
            T_opt = -V[init_id]
            results.append(dict(p=p, m=m, T_opt=T_opt))
            print(f"p={p:.2f} m={m} T_opt={T_opt:.4f}", flush=True)
    return results


def plot_Topt_vs_m(results, p_vals, m_vals, f0, delta,
                    outfile="figures/Topt_vs_m.png"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for p in p_vals:
        Ts = [r["T_opt"] for r in results if r["p"] == p]
        ax.plot(m_vals, Ts, marker="o", label=f"$p={p}$")
    ax.set_xlabel("$m$ (memories per node)")
    ax.set_ylabel(r"$T_{\mathrm{opt}}$")
    ax.set_title(f"Optimal expected waiting time vs. $m$  "
                 f"[$f_0={f0}$, " + r"$\Delta f$" + f"$={delta}$]")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


# =========================================================
# -----MAIN-----
# =========================================================
if __name__ == "__main__":
    os.makedirs("data/results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # ---- (p, f0) sweep ----
    m, delta = 4, 0.04
    p_vals = [0.3, 0.5, 0.7, 0.9]
    f0_vals = [0.62, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    results_pf0 = sweep_pf0(p_vals, f0_vals, m, delta)
    with open("data/results/advantage_grid.json", "w") as fp:
        json.dump(results_pf0, fp, indent=2)

    plot_per_baseline_heatmaps(results_pf0, p_vals, f0_vals, m, delta)
    plot_advantage_min(results_pf0, p_vals, f0_vals, m, delta)

    # ---- m sweep ----
    p_vals_m = [0.3, 0.6, 0.9]
    m_vals = [2, 3, 4, 5, 6]
    f0_m, delta_m = 0.75, 0.04

    results_m = sweep_m(p_vals_m, m_vals, f0_m, delta_m)
    with open("data/results/Topt_vs_m.json", "w") as fp:
        json.dump(results_m, fp, indent=2)

    plot_Topt_vs_m(results_m, p_vals_m, m_vals, f0_m, delta_m)
