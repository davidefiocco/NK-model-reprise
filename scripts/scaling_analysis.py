#!/usr/bin/env python3
"""
Cross-N Scaling Analysis

Combines phase diagram data across all system sizes to analyze:
1. Complete multi-N phase diagram with critical point scaling
2. Comparison of order parameter definitions (fraction vs mean d/N)
3. Rescaled data collapse (additive and multiplicative)
4. Saturation analysis with error bands

REQUIRES: Pre-computed trajectories in results/trajectories/ for multiple N values.
          Run analyze_trajectories.py first to verify per-N data.

Usage:
    python scripts/scaling_analysis.py                    # All figures
    python scripts/scaling_analysis.py --figure complete   # Just the 4-panel scaling figure
    python scripts/scaling_analysis.py --figure comparison  # Order parameter comparison
    python scripts/scaling_analysis.py --figure collapse    # Rescaled data collapse
    python scripts/scaling_analysis.py --figure saturation  # Saturation analysis
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional

# Directories
RESULTS_DIR = Path(__file__).parent.parent / "results" / "trajectories"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# System sizes and their plot properties
SIZE_COLORS = {
    20: '#e74c3c',    # red
    40: '#3498db',    # blue
    80: '#2ecc71',    # green
    160: '#9b59b6',   # purple
    320: '#f39c12',   # orange
}
SIZE_MARKERS = {20: 'o', 40: 's', 80: '^', 160: 'D', 320: 'p'}

# Threshold for "deep diffusing" classification
DEEP_DIFFUSING_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def detect_equilibration_auto(energy_traj: np.ndarray, gamma_acc_traj: np.ndarray,
                               window_frac: float = 0.15) -> float:
    """Auto-detect equilibration from energy trajectory (same as analyze_trajectories.py)."""
    n_points = len(energy_traj)
    if n_points < 20:
        return gamma_acc_traj[n_points // 2]

    window = max(5, int(n_points * window_frac))

    derivatives = []
    for i in range(window, n_points - window):
        x = gamma_acc_traj[i - window:i + window]
        y = energy_traj[i - window:i + window]
        if len(x) > 1 and (x[-1] - x[0]) > 0:
            slope = (y[-1] - y[0]) / (x[-1] - x[0])
            derivatives.append((gamma_acc_traj[i], abs(slope)))

    if not derivatives:
        return gamma_acc_traj[int(n_points * 0.5)]

    gammas, derivs = zip(*derivatives)
    derivs = np.array(derivs)
    threshold = 0.1 * np.max(derivs) if np.max(derivs) > 0 else 1e-6

    for i, d in enumerate(derivs):
        if d < threshold:
            if np.mean(derivs[i:] < threshold) > 0.7:
                return gammas[i]

    return gamma_acc_traj[int(n_points * 0.5)]


def load_phase_data(n: int, temperature: float = 0.6) -> pd.DataFrame:
    """
    Load trajectory data for a given N and compute steady-state statistics.

    Returns DataFrame with columns:
        gamma_max, d_mean, d_std, d_sem, n_samples, deep_fraction
    """
    data_dir = RESULTS_DIR / f"N{n:03d}"
    if not data_dir.exists():
        raise FileNotFoundError(f"No data directory for N={n}: {data_dir}")

    results = []

    for f in sorted(data_dir.glob("gamma_max_*.parquet")):
        gamma_max = float(f.stem.replace("gamma_max_", ""))
        df = pd.read_parquet(f)
        df_t = df[df['temperature'] == temperature]

        if df_t.empty:
            continue

        d_values = []
        for _, row in df_t.iterrows():
            gamma_acc = np.array(row['gamma_acc'])
            d = np.array(row['d'])
            energy = np.array(row['energy'])

            gamma_eq = detect_equilibration_auto(energy, gamma_acc)
            mask = gamma_acc > gamma_eq
            if np.any(mask):
                d_values.append(np.mean(d[mask]))

        if d_values:
            d_arr = np.array(d_values)
            results.append({
                'gamma_max': gamma_max,
                'd_mean': np.mean(d_arr),
                'd_std': np.std(d_arr),
                'd_sem': np.std(d_arr) / np.sqrt(len(d_arr)),
                'n_samples': len(d_arr),
                'deep_fraction': np.mean(d_arr >= DEEP_DIFFUSING_THRESHOLD),
            })

    return pd.DataFrame(results).sort_values('gamma_max').reset_index(drop=True)


def load_all_sizes(temperature: float = 0.6) -> Dict[int, pd.DataFrame]:
    """Load phase data for all available system sizes."""
    data = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith('N'):
            n = int(d.name[1:])
            try:
                df = load_phase_data(n, temperature)
                if not df.empty:
                    data[n] = df
                    print(f"  N={n}: {len(df)} gamma_max points, "
                          f"{df['n_samples'].iloc[0]} disorders")
            except Exception as e:
                print(f"  N={n}: skipped ({e})")
    return data


# ---------------------------------------------------------------------------
# Critical point estimation
# ---------------------------------------------------------------------------

def estimate_gamma_c_fraction(df: pd.DataFrame, threshold: float = 0.5) -> float:
    """
    Estimate gamma_c as the gamma_max where deep_fraction crosses a threshold.
    Uses linear interpolation.
    """
    gm = df['gamma_max'].values
    frac = df['deep_fraction'].values

    if frac[-1] < threshold or frac[0] > threshold:
        # Can't interpolate -- return midpoint of range
        return (gm[0] + gm[-1]) / 2

    f = interp1d(frac, gm, kind='linear', bounds_error=False, fill_value='extrapolate')
    return float(f(threshold))


def estimate_gamma_c_mean(df: pd.DataFrame, threshold: float = 0.10) -> float:
    """
    Estimate gamma_c as the gamma_max where mean d/N crosses a threshold.
    Uses linear interpolation.
    """
    gm = df['gamma_max'].values
    d_mean = df['d_mean'].values

    if d_mean[-1] < threshold or d_mean[0] > threshold:
        return (gm[0] + gm[-1]) / 2

    f = interp1d(d_mean, gm, kind='linear', bounds_error=False, fill_value='extrapolate')
    return float(f(threshold))


def power_law(n, a, b):
    """Power law: gamma_c = a * N^b"""
    return a * np.power(n, b)


def inverse_n(n, a, b):
    """1/N form: gamma_c = a + b/N"""
    return a + b / n


# ---------------------------------------------------------------------------
# Figure 1: Complete Analysis (4-panel)
# ---------------------------------------------------------------------------

def plot_complete_analysis(data: Dict[int, pd.DataFrame], save: bool = True):
    """
    4-panel figure:
    - Top-left: Phase diagram (all N, deep diffusing fraction vs gamma_max)
    - Top-right: Zoom on transition region (large N only)
    - Bottom-left: gamma_c vs N with power law and 1/N fits
    - Bottom-right: Log-log plot with linear fit
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Compute gamma_c for each N
    sizes = sorted(data.keys())
    gamma_c_values = {}
    for n in sizes:
        gamma_c_values[n] = estimate_gamma_c_fraction(data[n])

    # --- Top-left: Phase diagram, all N ---
    ax = axes[0, 0]
    for n in sizes:
        df = data[n]
        ax.plot(df['gamma_max'], df['deep_fraction'],
                marker=SIZE_MARKERS[n], color=SIZE_COLORS[n],
                label=f'N={n}', linewidth=1.5, markersize=6)
    ax.set_xlabel(r'$\gamma_{\max}$', fontsize=12)
    ax.set_ylabel(r'Deep Diffusing Fraction ($d/N \geq 0.20$)', fontsize=11)
    ax.set_title('Phase Diagram: All System Sizes', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Top-right: Zoom on transition (N >= 80) ---
    ax = axes[0, 1]
    large_sizes = [n for n in sizes if n >= 80]
    for n in large_sizes:
        df = data[n]
        ax.plot(df['gamma_max'], df['deep_fraction'],
                marker=SIZE_MARKERS[n], color=SIZE_COLORS[n],
                label=f'N={n}', linewidth=1.5, markersize=6)
    ax.set_xlabel(r'$\gamma_{\max}$', fontsize=12)
    ax.set_ylabel(r'Deep Diffusing Fraction', fontsize=11)
    ax.set_title('Zoom: Transition Region', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: gamma_c vs N ---
    ax = axes[1, 0]
    ns = np.array(sizes, dtype=float)
    gcs = np.array([gamma_c_values[n] for n in sizes])

    ax.scatter(ns, gcs, c=[SIZE_COLORS[n] for n in sizes],
               s=120, zorder=5, edgecolors='black', linewidths=0.5)
    for n, gc in zip(sizes, gcs):
        ax.annotate(f'N={n}', (n, gc), textcoords="offset points",
                    xytext=(8, 8), fontsize=9)

    # Fit power law
    try:
        popt_pow, _ = curve_fit(power_law, ns, gcs, p0=[3.0, -0.5], maxfev=5000)
        n_fit = np.linspace(min(ns) * 0.8, max(ns) * 1.2, 200)
        ax.plot(n_fit, power_law(n_fit, *popt_pow), '-',
                color='#2ecc71', linewidth=2,
                label=f'Power law: {popt_pow[0]:.1f}$\\cdot N^{{{popt_pow[1]:.2f}}}$')
        rss_pow = np.sum((gcs - power_law(ns, *popt_pow)) ** 2)
    except RuntimeError:
        popt_pow = None
        rss_pow = np.inf

    # Fit 1/N
    try:
        popt_inv, _ = curve_fit(inverse_n, ns, gcs, p0=[0.3, 10], maxfev=5000)
        ax.plot(n_fit, inverse_n(n_fit, *popt_inv), '--',
                color='#e74c3c', linewidth=2,
                label=f'1/N: {popt_inv[0]:.2f} + {popt_inv[1]:.0f}/N')
        rss_inv = np.sum((gcs - inverse_n(ns, *popt_inv)) ** 2)
    except RuntimeError:
        popt_inv = None
        rss_inv = np.inf

    ax.set_xlabel('N', fontsize=12)
    ax.set_ylabel(r'$\gamma_c$', fontsize=12)
    ax.set_title('Critical Point Scaling', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Bottom-right: Log-log ---
    ax = axes[1, 1]
    ln_n = np.log(ns)
    ln_gc = np.log(gcs)

    ax.scatter(ln_n, ln_gc, c=[SIZE_COLORS[n] for n in sizes],
               s=120, zorder=5, edgecolors='black', linewidths=0.5)
    for n, lx, ly in zip(sizes, ln_n, ln_gc):
        ax.annotate(f'N={n}', (lx, ly), textcoords="offset points",
                    xytext=(8, 8), fontsize=9)

    # Linear fit in log-log
    slope, intercept = np.polyfit(ln_n, ln_gc, 1)
    ln_n_fit = np.linspace(ln_n.min() - 0.2, ln_n.max() + 0.2, 100)
    ax.plot(ln_n_fit, slope * ln_n_fit + intercept, '-',
            color='#2ecc71', linewidth=2,
            label=f'Linear fit slope: {slope:.2f}')

    ax.set_xlabel(r'$\ln(N)$', fontsize=12)
    ax.set_ylabel(r'$\ln(\gamma_c)$', fontsize=12)
    ax.set_title('Log-Log Plot', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'NK Model: Complete Analysis with N={max(sizes)} '
        f'({len(sizes)} points)\n'
        f'Power Law RSS={rss_pow:.5f} vs 1/N RSS={rss_inv:.5f}',
        fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save:
        out = FIGURES_DIR / 'complete_analysis_final.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Figure 2: Order Parameter Comparison (4-panel)
# ---------------------------------------------------------------------------

def plot_order_parameter_comparison(data: Dict[int, pd.DataFrame], save: bool = True):
    """
    Compare two order parameter definitions:
    - Deep diffusing fraction (d/N >= 0.20)
    - Mean d/N
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    sizes = sorted(data.keys())

    gamma_c_frac = {}
    gamma_c_mean = {}
    for n in sizes:
        gamma_c_frac[n] = estimate_gamma_c_fraction(data[n])
        gamma_c_mean[n] = estimate_gamma_c_mean(data[n])

    # --- Top-left: Fraction-based ---
    ax = axes[0, 0]
    for n in sizes:
        df = data[n]
        ax.plot(df['gamma_max'], df['deep_fraction'],
                marker=SIZE_MARKERS[n], color=SIZE_COLORS[n],
                label=f'N={n}', linewidth=1.5, markersize=6)
    ax.set_xlabel(r'$\gamma_{\max}$', fontsize=12)
    ax.set_ylabel(r'Fraction with $d/N \geq 0.20$', fontsize=11)
    ax.set_title(r'Order Parameter: Fraction with $d/N \geq 0.20$', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Top-right: Mean d/N ---
    ax = axes[0, 1]
    for n in sizes:
        df = data[n]
        ax.plot(df['gamma_max'], df['d_mean'],
                marker=SIZE_MARKERS[n], color=SIZE_COLORS[n],
                label=f'N={n}', linewidth=1.5, markersize=6)
    ax.set_xlabel(r'$\gamma_{\max}$', fontsize=12)
    ax.set_ylabel(r'$\langle d/N \rangle$', fontsize=12)
    ax.set_title(r'Order Parameter: $\langle d/N \rangle$', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: gamma_c comparison ---
    ax = axes[1, 0]
    ns = np.array(sizes, dtype=float)
    gc_f = np.array([gamma_c_frac[n] for n in sizes])
    gc_m = np.array([gamma_c_mean[n] for n in sizes])

    ax.scatter(ns, gc_f, c='#3498db', s=100, marker='o', label='From fraction', zorder=5)
    ax.scatter(ns, gc_m, c='#e74c3c', s=100, marker='s', label=r'From $\langle d/N \rangle$', zorder=5)

    # Fit power laws
    n_fit = np.linspace(min(ns) * 0.8, max(ns) * 1.2, 200)
    try:
        popt_f, _ = curve_fit(power_law, ns, gc_f, p0=[3.0, -0.5])
        ax.plot(n_fit, power_law(n_fit, *popt_f), '-', color='#3498db', linewidth=2)
    except RuntimeError:
        popt_f = [np.nan, np.nan]
    try:
        popt_m, _ = curve_fit(power_law, ns, gc_m, p0=[3.0, -0.5])
        ax.plot(n_fit, power_law(n_fit, *popt_m), '--', color='#e74c3c', linewidth=2)
    except RuntimeError:
        popt_m = [np.nan, np.nan]

    ax.set_xlabel('N', fontsize=12)
    ax.set_ylabel(r'$\gamma_c$', fontsize=12)
    ax.set_title('Critical Point Comparison', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Bottom-right: Log-log both ---
    ax = axes[1, 1]
    ln_n = np.log(ns)

    slope_f, intercept_f = np.polyfit(ln_n, np.log(gc_f), 1)
    slope_m, intercept_m = np.polyfit(ln_n, np.log(gc_m), 1)

    ax.scatter(ln_n, np.log(gc_f), c='#3498db', s=100, marker='o', zorder=5)
    ax.scatter(ln_n, np.log(gc_m), c='#e74c3c', s=100, marker='s', zorder=5)

    ln_n_fit = np.linspace(ln_n.min() - 0.2, ln_n.max() + 0.2, 100)
    ax.plot(ln_n_fit, slope_f * ln_n_fit + intercept_f, '-',
            color='#3498db', linewidth=2,
            label=f'Fraction: slope={slope_f:.2f}')
    ax.plot(ln_n_fit, slope_m * ln_n_fit + intercept_m, '--',
            color='#e74c3c', linewidth=2,
            label=rf'$\langle d/N \rangle$: slope={slope_m:.2f}')

    ax.set_xlabel(r'$\ln(N)$', fontsize=12)
    ax.set_ylabel(r'$\ln(\gamma_c)$', fontsize=12)
    ax.set_title('Log-Log: Both Order Parameters', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Order Parameter Comparison: Deep Fraction vs Mean Hamming Distance',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save:
        out = FIGURES_DIR / 'order_parameter_comparison.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Figure 3: Rescaled Data Collapse (3-panel)
# ---------------------------------------------------------------------------

def plot_rescaled_collapse(data: Dict[int, pd.DataFrame], save: bool = True):
    """
    3-panel figure testing data collapse:
    - Raw: d/N vs gamma_max
    - Additive: d/N vs (gamma_max - gamma_c)
    - Multiplicative: d/N vs (gamma_max / gamma_c)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sizes = sorted(data.keys())

    gamma_c_values = {}
    for n in sizes:
        gamma_c_values[n] = estimate_gamma_c_mean(data[n])

    # --- Left: Raw ---
    ax = axes[0]
    for n in sizes:
        df = data[n]
        ax.plot(df['gamma_max'], df['d_mean'],
                marker=SIZE_MARKERS[n], color=SIZE_COLORS[n],
                label=f'N={n}', linewidth=1.5, markersize=5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, label='Full decorrelation')
    ax.set_xlabel(r'$\gamma_{\max}$', fontsize=12)
    ax.set_ylabel(r'$\langle d/N \rangle$', fontsize=12)
    ax.set_title(r'Raw: $\langle d/N \rangle$ vs $\gamma_{\max}$', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Middle: Additive rescaling ---
    ax = axes[1]
    for n in sizes:
        df = data[n]
        gc = gamma_c_values[n]
        ax.plot(df['gamma_max'] - gc, df['d_mean'],
                marker=SIZE_MARKERS[n], color=SIZE_COLORS[n],
                label=f'N={n}', linewidth=1.5, markersize=5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlabel(r'$\gamma_{\max} - \gamma_c$  (distance from transition)', fontsize=11)
    ax.set_ylabel(r'$\langle d/N \rangle$', fontsize=12)
    ax.set_title(r'Rescaled: $\langle d/N \rangle$ vs $(\gamma_{\max} - \gamma_c)$', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Right: Multiplicative rescaling ---
    ax = axes[2]
    for n in sizes:
        df = data[n]
        gc = gamma_c_values[n]
        ax.plot(df['gamma_max'] / gc, df['d_mean'],
                marker=SIZE_MARKERS[n], color=SIZE_COLORS[n],
                label=f'N={n}', linewidth=1.5, markersize=5)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5,
               label=r'$\gamma_{\max} = \gamma_c$')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlabel(r'$\gamma_{\max} / \gamma_c$', fontsize=12)
    ax.set_ylabel(r'$\langle d/N \rangle$', fontsize=12)
    ax.set_title(r'Multiplicative: $\langle d/N \rangle$ vs $(\gamma_{\max} / \gamma_c)$', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        out = FIGURES_DIR / 'rescaled_decorrelation.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Figure 4: Saturation Analysis (2-panel)
# ---------------------------------------------------------------------------

def plot_saturation_analysis(data: Dict[int, pd.DataFrame], save: bool = True):
    """
    2-panel figure:
    - Left: Mean d/N vs gamma_max with +/- 1 sigma error bands
    - Right: Zoom on low gamma_max region (N=320 range)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sizes = sorted(data.keys())

    max_n = max(sizes)
    max_n_range = data[max_n]['gamma_max'].max()

    # --- Left: Full range with error bands ---
    ax = axes[0]
    for n in sizes:
        df = data[n]
        ax.plot(df['gamma_max'], df['d_mean'],
                marker=SIZE_MARKERS[n], color=SIZE_COLORS[n],
                label=f'N={n}', linewidth=1.5, markersize=5, zorder=3)
        ax.fill_between(df['gamma_max'],
                        df['d_mean'] - df['d_std'],
                        df['d_mean'] + df['d_std'],
                        color=SIZE_COLORS[n], alpha=0.15)
    ax.axvline(x=max_n_range, color=SIZE_COLORS[max_n], linestyle='--', alpha=0.5,
               label=f'N={max_n} max range')
    ax.set_xlabel(r'$\gamma_{\max}$', fontsize=12)
    ax.set_ylabel(r'$\langle d/N \rangle$', fontsize=12)
    ax.set_title(r'Mean Hamming Distance vs $\gamma_{\max}$' + '\n(shaded = $\\pm 1\\sigma$)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Right: Zoom on small gamma_max ---
    ax = axes[1]
    zoom_max = max_n_range + 0.05
    for n in sizes:
        df = data[n]
        df_zoom = df[df['gamma_max'] <= zoom_max]
        if not df_zoom.empty:
            ax.plot(df_zoom['gamma_max'], df_zoom['d_mean'],
                    marker=SIZE_MARKERS[n], color=SIZE_COLORS[n],
                    label=f'N={n}', linewidth=1.5, markersize=6)
    ax.set_xlabel(r'$\gamma_{\max}$', fontsize=12)
    ax.set_ylabel(r'$\langle d/N \rangle$', fontsize=12)
    ax.set_title(rf'Zoom: $\gamma_{{\max}} \leq {zoom_max:.2f}$ (N={max_n} range)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        out = FIGURES_DIR / 'saturation_analysis.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='NK Model Cross-N Scaling Analysis')
    parser.add_argument('--figure', type=str, default='all',
                        choices=['all', 'complete', 'comparison', 'collapse', 'saturation'],
                        help='Which figure to generate (default: all)')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature to analyze (default: 0.6)')
    parser.add_argument('--no-save', action='store_true',
                        help='Show plots interactively instead of saving')
    args = parser.parse_args()

    save = not args.no_save
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading phase data for T={args.temperature}...")
    data = load_all_sizes(args.temperature)

    if len(data) < 2:
        print("Need at least 2 system sizes for scaling analysis.")
        return

    print(f"\nSystem sizes available: {sorted(data.keys())}")

    # Print gamma_c estimates
    print("\nCritical point estimates (fraction-based):")
    for n in sorted(data.keys()):
        gc = estimate_gamma_c_fraction(data[n])
        print(f"  N={n:4d}: gamma_c = {gc:.3f}")

    print("\nCritical point estimates (mean d/N):")
    for n in sorted(data.keys()):
        gc = estimate_gamma_c_mean(data[n])
        print(f"  N={n:4d}: gamma_c = {gc:.3f}")

    figures = {
        'complete': plot_complete_analysis,
        'comparison': plot_order_parameter_comparison,
        'collapse': plot_rescaled_collapse,
        'saturation': plot_saturation_analysis,
    }

    if args.figure == 'all':
        for name, func in figures.items():
            print(f"\nGenerating {name}...")
            func(data, save=save)
    else:
        print(f"\nGenerating {args.figure}...")
        figures[args.figure](data, save=save)

    print("\nDone!")


if __name__ == '__main__':
    main()
