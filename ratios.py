#!/usr/bin/env python3
"""
band_ratios.py

Compute alpha/delta and beta/theta power ratios from EEG time-series files.

Usage:
  python band_ratios.py file1.txt file2.txt ... [--out ratios.csv]

Input format:
  Each file: two columns -> time (seconds), eeg (mV), separated by whitespace or commas.

Method:
  - Estimate sampling rate from the time column.
  - Compute one-sided PSD (periodogram, Hamming, density).
  - Integrate power in bands:
      delta: 0.5–4 Hz
      theta: 4–8 Hz
      alpha: 8–13 Hz
      beta : 13–30 Hz
  - Ratios: alpha/delta, beta/theta
"""

import argparse
import os
import numpy as np
from scipy import signal
import csv
from textwrap import shorten

# Band definitions (Hz)
DELTA = (0.5, 4.0)
THETA = (4.0, 8.0)
ALPHA = (8.0, 13.0)
BETA  = (13.0, 30.0)

def load_time_series(path):
    # Try whitespace first, then comma
    try:
        arr = np.loadtxt(path)
        if arr.ndim != 2 or arr.shape[1] < 2:
            arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = np.loadtxt(path, delimiter=",")
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"'{path}': expected 2 columns (time, eeg)")
    t = arr[:, 0].astype(float)
    x = arr[:, 1].astype(float)
    return t, x

def estimate_fs(t):
    dt = np.diff(t)
    if len(dt) == 0 or not np.all(np.isfinite(dt)):
        raise RuntimeError("Cannot estimate sampling rate from time column.")
    return 1.0 / np.median(dt)

def compute_psd(x, fs):
    # Remove DC to avoid a big spike near 0 Hz
    x = np.asarray(x) - np.mean(x)
    f, Pxx = signal.periodogram(
        x, fs=fs, window="hamming", detrend=False,
        scaling="density", return_onesided=True
    )
    return f, Pxx  # Pxx units: (mV^2/Hz)

def band_power(f, Pxx, lo, hi):
    lo2, hi2 = min(lo, hi), max(lo, hi)
    # Clip to valid positive freqs within Nyquist
    mask = (f >= lo2) & (f <= hi2)
    if not np.any(mask):
        return 0.0
    # Integrate PSD (density) over frequency to get power
    return float(np.trapz(Pxx[mask], f[mask]))

def process_file(path):
    t, x = load_time_series(path)
    fs = estimate_fs(t)
    f, Pxx = compute_psd(x, fs)

    p_delta = band_power(f, Pxx, *DELTA)
    p_theta = band_power(f, Pxx, *THETA)
    p_alpha = band_power(f, Pxx, *ALPHA)
    p_beta  = band_power(f, Pxx, *BETA)

    alpha_over_delta = (p_alpha / p_delta) if p_delta > 0 else np.nan
    beta_over_theta  = (p_beta  / p_theta) if p_theta > 0 else np.nan

    return {
        "file": os.path.basename(path),
        "fs_Hz": fs,
        "P_delta": p_delta,
        "P_theta": p_theta,
        "P_alpha": p_alpha,
        "P_beta": p_beta,
        "alpha_over_delta": alpha_over_delta,
        "beta_over_theta": beta_over_theta,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="Input .txt files (time[s], EEG[mV])")
    ap.add_argument("--out", type=str, default=None, help="Optional CSV output path")
    args = ap.parse_args()

    results = []
    for path in args.files:
        try:
            results.append(process_file(path))
        except Exception as e:
            print(f"[ERROR] {os.path.basename(path)}: {e}")

    if results:
        # Console table
        header = ["File", "fs (Hz)", "α/δ", "β/θ", "Pδ", "Pθ", "Pα", "Pβ"]
        print(" | ".join(header))
        print("-" * 90)
        for r in results:
            print(" | ".join([
                shorten(r["file"], width=28, placeholder="…"),
                f"{r['fs_Hz']:.2f}",
                f"{r['alpha_over_delta']:.3f}" if np.isfinite(r["alpha_over_delta"]) else "nan",
                f"{r['beta_over_theta']:.3f}" if np.isfinite(r["beta_over_theta"]) else "nan",
                f"{r['P_delta']:.6e}",
                f"{r['P_theta']:.6e}",
                f"{r['P_alpha']:.6e}",
                f"{r['P_beta']:.6e}",
            ]))

        # CSV (optional)
        if args.out:
            try:
                with open(args.out, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "file","fs_Hz","P_delta","P_theta","P_alpha","P_beta",
                        "alpha_over_delta","beta_over_theta"
                    ])
                    writer.writeheader()
                    for r in results:
                        writer.writerow(r)
                print(f"\nSaved: {args.out}")
            except Exception as e:
                print(f"[ERROR] Could not write CSV '{args.out}': {e}")

if __name__ == "__main__":
    main()
