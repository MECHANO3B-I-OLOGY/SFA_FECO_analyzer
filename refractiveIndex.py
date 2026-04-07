"""
refractiveIndex.py

Computes the refractive index μ of the medium in an SFA experiment from two
FECO fringe CSV files (one odd fringe, one even fringe).

Usage:
    python refractiveIndex.py oddFringe.csv evenFringe.csv

Formula:
    μ = sqrt( (λD_{n-1} - λ0_{n-1}) * (n-1) * F_{n-1} /
              (λD_n    - λ0_n   ) * n     * F_n     ) * μ_mica

    where:
        F_n     = λ0_{n-1} / (λ0_{n-1} - λ0_n)
        F_{n-1} = λ0_{n-2} / (λ0_{n-2} - λ0_{n-1})
        μ_mica  = 1.5820 + 4760 / λD_n²        (λ in nm)

All wavelengths are in nanometers.
n must be an odd fringe order.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_fringe(path: str) -> pd.DataFrame:
    """Load a fringe CSV and return a DataFrame indexed by time."""
    df = pd.read_csv(path, usecols=["time", "wavelength"])
    df = df.drop_duplicates(subset="time").set_index("time").sort_index()
    return df


def prompt_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  Please enter a valid number.")


def prompt_int(prompt: str) -> int:
    while True:
        try:
            val = int(input(prompt).strip())
            return val
        except ValueError:
            print("  Please enter a valid integer.")


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def fringe_order_factor(lam0_lower: float, lam0_upper: float) -> float:
    """
    F_n = λ0_{n-1} / (λ0_{n-1} - λ0_n)

    where lam0_lower = λ0_{n-1}  (the *lower* order reference wavelength)
          lam0_upper = λ0_n      (the *higher* order reference wavelength)

    Note: λ0_{n-1} > λ0_n for typical FECO fringes (longer wavelength at
    lower order), so the denominator is positive.
    """
    denom = lam0_lower - lam0_upper
    if abs(denom) < 1e-12:
        raise ValueError(
            f"Reference wavelengths λ0={lam0_lower} and λ0={lam0_upper} "
            "are too close; cannot compute fringe order factor."
        )
    return lam0_lower / denom


def mu_mica(lam_D_n: np.ndarray) -> np.ndarray:
    """Cauchy dispersion for mica: μ_mica = 1.5820 + 4760 / λD_n²  (λ in nm)."""
    return 1.5820 + 4760.0 / lam_D_n**2


def compute_mu(
    lam_D_n: np.ndarray,      # measured wavelength of odd fringe n   (nm)
    lam_D_n1: np.ndarray,     # measured wavelength of even fringe n-1 (nm)
    lam0_n2: float,           # reference wavelength of fringe n-2
    lam0_n1: float,           # reference wavelength of fringe n-1
    lam0_n: float,            # reference wavelength of fringe n
    n: int,                   # fringe order (odd)
) -> np.ndarray:
    """
    μ = sqrt( [(λD_{n-1} - λ0_{n-1}) * (n-1) * F_{n-1}] /
               [(λD_n    - λ0_n   ) * n     * F_n    ] ) * μ_mica
    """
    F_n  = fringe_order_factor(lam0_n1, lam0_n)    # uses λ0_{n-1}, λ0_n
    F_n1 = fringe_order_factor(lam0_n2, lam0_n1)   # uses λ0_{n-2}, λ0_{n-1}

    numerator   = (lam_D_n1 - lam0_n1) * (n - 1) * F_n1
    denominator = (lam_D_n  - lam0_n ) *  n       * F_n

    ratio = numerator / denominator

    if np.any(ratio < 0):
        n_neg = np.sum(ratio < 0)
        print(
            f"\n  Warning: {n_neg} time point(s) produced a negative ratio "
            "inside the square root.\n"
            "  These will appear as NaN in the output. "
            "Check that fringe order assignments and reference wavelengths are correct.\n"
        )

    return np.sqrt(np.where(ratio >= 0, ratio, np.nan)) * mu_mica(lam_D_n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Command-line arguments -------------------------------------------
    if len(sys.argv) != 3:
        print("Usage: python refractiveIndex.py oddFringe.csv evenFringe.csv")
        sys.exit(1)

    odd_path, even_path = sys.argv[1], sys.argv[2]

    for p in (odd_path, even_path):
        if not os.path.isfile(p):
            print(f"Error: file not found: {p}")
            sys.exit(1)

    # --- Load data --------------------------------------------------------
    odd_df  = load_fringe(odd_path)
    even_df = load_fringe(even_path)

    # Intersect timestamps
    common_times = odd_df.index.intersection(even_df.index)
    if common_times.empty:
        print("Error: the two files share no common time stamps.")
        sys.exit(1)

    odd_shared  = odd_df.loc[common_times, "wavelength"].values
    even_shared = even_df.loc[common_times, "wavelength"].values

    t_min, t_max = common_times.min(), common_times.max()

    # --- User inputs ------------------------------------------------------
    print()
    n = prompt_int("Input fringe order n: ")
    if n % 2 == 0:
        print(
            f"  Warning: n={n} is even. The formula is defined for odd fringes only.\n"
            "  Proceeding, but results may not be physically meaningful."
        )

    print(f"Input fringe {n-2} reference wavelength (nm): ", end="")
    lam0_n2 = prompt_float("")

    print(f"Input fringe {n-1} reference wavelength (nm): ", end="")
    lam0_n1 = prompt_float("")

    print(f"Input fringe {n} reference wavelength (nm): ", end="")
    lam0_n  = prompt_float("")

    # --- Computation ------------------------------------------------------
    print(f"\nComputing μ over {len(common_times)} shared time points "
          f"(t = {t_min} - {t_max})...\n")

    mu_values = compute_mu(
        lam_D_n  = odd_shared,
        lam_D_n1 = even_shared, # HERE: + 10 for testing
        lam0_n2  = lam0_n2,
        lam0_n1  = lam0_n1,
        lam0_n   = lam0_n,
        n        = n,
    )

    result_df = pd.DataFrame({
        "time":             common_times,
        "refractive_index": mu_values,
    })

    valid = result_df["refractive_index"].notna().sum()
    print(f"  Valid results : {valid} / {len(result_df)}")
    print(f"  μ range       : {np.nanmin(mu_values):.6f} - {np.nanmax(mu_values):.6f}")
    print(f"  μ mean        : {np.nanmean(mu_values):.6f}")

    # --- Save output ------------------------------------------------------
    out_dir = "Output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "refractiveIndex.csv")

    result_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # --- Plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        result_df["time"],
        result_df["refractive_index"],
        color="steelblue",
        linewidth=1.5,
        label=f"μ  (fringe {n})",
    )
    ax.scatter(
        result_df["time"],
        result_df["refractive_index"],
        s=18,
        color="steelblue",
        zorder=3,
    )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Refractive index  μ", fontsize=12)
    ax.set_title(
        f"Refractive index of medium  —  fringe {n} (odd) & {n-1} (even)\n"
        f"λ⁰_{{n-2}}={lam0_n2} nm,  λ⁰_{{n-1}}={lam0_n1} nm,  λ⁰_n={lam0_n} nm",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
