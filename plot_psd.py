import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import sys
import os

def plot_psd(files, duration=50, fmin=0, fmax=35):
    """
    Compute and plot EEG Power Spectral Density (PSD) for multiple files
    using a Hamming window, limited to 0.5–40 Hz, first 50 seconds.
    Also computes average band power (δ, θ, α, β).
    """

    # Define EEG bands
    bands = {
        "Delta (0.5–4 Hz)": (0.5, 4),
        "Theta (4–8 Hz)": (4, 8),
        "Alpha (8–13 Hz)": (8, 13),
        "Beta (13–30 Hz)": (13, 30)
    }

    plt.figure(figsize=(10, 6))
    band_power_summary = []

    for file in files:
        # Load time and voltage
        time, mv = np.loadtxt(file, unpack=True)

        # Auto-detect sampling frequency
        fs = 1 / np.mean(np.diff(time))

        # Restrict to first 'duration' seconds
        mask = time <= duration
        mv = mv[mask]

        # Hamming window for Welch’s method
        window = np.hamming(int(fs * 2))  # 2-second window

        # Compute PSD
        f, Pxx = welch(mv, fs=fs, window=window, nperseg=len(window))

        # Limit frequency range to 0.5–40 Hz
        band_mask = (f >= fmin) & (f <= fmax)
        f_band = f[band_mask]
        Pxx_band = Pxx[band_mask]

        # Compute average band powers
        band_powers = {}
        for name, (low, high) in bands.items():
            mask_band = (f >= low) & (f < high)
            band_powers[name] = np.trapz(Pxx[mask_band], f[mask_band])

        band_power_summary.append((os.path.basename(file), band_powers))

        # Plot PSD
        plt.plot(f_band, Pxx_band, label=os.path.basename(file), alpha=0.8)

    # Plot formatting
    plt.title(f'EEG Power Spectral Density (Hamming, {fmin}-{fmax} Hz, First {duration}s)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (mV²/Hz)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print average power summary
    print("\nAverage Power in EEG Bands (mV²):")
    print("=" * 60)
    for fname, bp in band_power_summary:
        print(f"\n{fname}:")
        for band_name, value in bp.items():
            print(f"  {band_name:<18} {value:.4e}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_psd.py <file1> <file2> ...")
        sys.exit(1)

    files = sys.argv[1:]
    plot_psd(files)
