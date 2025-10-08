#!/usr/bin/env python3
"""
psd_features.py

Usage:
  python psd_features.py file1.txt file2.txt ... [--fmin 0.5 --fmax 40 --out summary.csv]

Each input file must have two whitespace- or comma-separated columns:
time (s), eeg (mV)

Outputs:
- A plot overlaying the PSDs for all files (linear axes, not log)
- Console table of Mean Power (band-averaged), Median Frequency, Peak Frequency within [fmin, fmax]
- Optional CSV with the same summary if --out is provided
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import csv
from textwrap import shorten

def load_time_series(path):
    try:
        arr = np.loadtxt(path, delimiter=None)  # auto-detect whitespace
        if arr.ndim != 2 or arr.shape[1] < 2:
            # try comma explicitly if needed
            arr = np.loadtxt(path, delimiter=",")
        t = arr[:, 0].astype(float)
        x = arr[:, 1].astype(float)
        return t, x
    except Exception as e:
        raise RuntimeError(f"Failed to read '{path}': {e}")

def estimate_fs(t):
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)) or len(dt) == 0:
        raise RuntimeError("Cannot estimate sampling rate from time column.")
    # Use median to be robust to a stray timestamp
    fs = 1.0 / np.median(dt)
    return fs

def band_indices(f, fmin, fmax):
    return (f >= fmin) & (f <= fmax)

def band_mean_power(f, Pxx, fmin, fmax):
    """Band-AVERAGED power (integral of PSD over band / bandwidth)."""
    m = band_indices(f, fmin, fmax)
    if not np.any(m):
        return np.nan
    band_power = np.trapz(Pxx[m], f[m])  # total power in band
    bw = fmax - fmin
    return band_power / bw if bw > 0 else np.nan

def band_median_frequency(f, Pxx, fmin, fmax):
    """Frequency where cumulative band power reaches 50%."""
    m = band_indices(f, fmin, fmax)
    if not np.any(m):
        return np.nan
    f_band = f[m]
    P_band = Pxx[m]
    c = np.cumsum((P_band[:-1] + P_band[1:]) / 2 * np.diff(f_band))  # cumulative integral via trapezoid
    total = c[-1] if len(c) > 0 else 0.0
    if total <= 0:
        return np.nan
    target = 0.5 * total
    # Find first index where cumulative >= target
    idx = np.searchsorted(c, target)
    # Map back to a frequency (between points). Use linear interpolation across the interval.
    if idx == 0:
        return f_band[0]
    if idx >= len(c):
        return f_band[-1]
    # cumulative at idx-1 and idx corresponds to interval between f_band[idx] and f_band[idx+1]
    c0 = c[idx - 1]
    c1 = c[idx]
    f0 = f_band[idx]
    f1 = f_band[idx + 1]
    if c1 == c0:
        return f0
    alpha = (target - c0) / (c1 - c0)
    return f0 + alpha * (f1 - f0)

def band_peak_frequency(f, Pxx, fmin, fmax):
    m = band_indices(f, fmin, fmax)
    if not np.any(m):
        return np.nan
    idx_rel = np.argmax(Pxx[m])
    return f[m][idx_rel]

def compute_psd_full(x, fs):
    """
    PSD over the entire recording using periodogram with a Hamming window.
    We detrend by removing the mean first to avoid a huge DC spike.
    """
    x = np.asarray(x)
    x = x - np.mean(x)
    f, Pxx = signal.periodogram(
        x,
        fs=fs,
        window="hamming",
        detrend=False,
        scaling="density",
        nfft=None,  # default power of two >= len(x)
        return_onesided=True
    )
    return f, Pxx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="Input .txt files with time(s) and EEG(mV)")
    ap.add_argument("--fmin", type=float, default=0.5, help="Lower frequency bound (Hz)")
    ap.add_argument("--fmax", type=float, default=40.0, help="Upper frequency bound (Hz)")
    ap.add_argument("--out", type=str, default=None, help="Optional CSV path to save summary")
    args = ap.parse_args()

    summaries = []
    plt.figure()
    for path in args.files:
        label = os.path.basename(path)
        try:
            t, x = load_time_series(path)
            fs = estimate_fs(t)
            f, Pxx = compute_psd_full(x, fs)

            # plot band-limited PSD
            m = band_indices(f, args.fmin, args.fmax)
            if not np.any(m):
                print(f"[WARN] No frequencies within [{args.fmin}, {args.fmax}] Hz for {label}")
                continue

            plt.plot(f[m], Pxx[m], label=label)

            mean_p = band_mean_power(f, Pxx, args.fmin, args.fmax)
            med_f = band_median_frequency(f, Pxx, args.fmin, args.fmax)
            peak_f = band_peak_frequency(f, Pxx, args.fmin, args.fmax)

            summaries.append({
                "file": label,
                "fs_Hz": fs,
                "mean_power_bandavg": mean_p,
                "median_freq_Hz": med_f,
                "peak_freq_Hz": peak_f
            })
        except Exception as e:
            print(f"[ERROR] {label}: {e}")

    # Finish plot
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (mV^2/Hz)")
    plt.title(f"EEG PSD (Hamming, {args.fmin}-{args.fmax} Hz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print a console table
    if len(summaries) > 0:
        header = ["File", "fs (Hz)", "Mean P (band-avg)", "Median F (Hz)", "Peak F (Hz)"]
        row_fmt = "{:<28} {:>9.2f} {:>18.6e} {:>14.3f} {:>12.3f}"
        print("\n" + " | ".join(header))
        print("-" * 80)
        for s in summaries:
            print(row_fmt.format(
                shorten(s["file"], width=28, placeholder="…"),
                s["fs_Hz"],
                s["mean_power_bandavg"],
                s["median_freq_Hz"],
                s["peak_freq_Hz"],
            ))

    # Optional CSV
    if args.out and len(summaries) > 0:
        try:
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["file","fs_Hz","mean_power_bandavg","median_freq_Hz","peak_freq_Hz"])
                w.writeheader()
                w.writerows(summaries)
            print(f"\nSaved summary to: {args.out}")
        except Exception as e:
            print(f"[ERROR] Could not save CSV '{args.out}': {e}")

if __name__ == "__main__":
    main()
