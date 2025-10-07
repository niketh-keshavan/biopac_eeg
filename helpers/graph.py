import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_eeg(file1, file2):
    # Load both files (assuming two columns: time, mV)
    time1, mv1 = np.loadtxt(file1, unpack=True)
    time2, mv2 = np.loadtxt(file2, unpack=True)

    # Create the plots
    plt.figure(figsize=(10, 6))
    plt.plot(time1, mv1, label=f'File 1: {file1}', alpha=0.8)
    plt.plot(time2, mv2, label=f'File 2: {file2}', alpha=0.8)

    # Labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("EEG (mV)")
    plt.title("EEG Signals Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_eeg.py <file1> <file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    plot_eeg(file1, file2)
