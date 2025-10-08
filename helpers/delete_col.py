#!/usr/bin/env python3
"""
delete_columns.py

Removes the 3rd and 4th columns from a text file
and writes the result to a new file (default: *_trimmed.txt).
"""

import sys
import os

if len(sys.argv) < 2:
    print("Usage: python delete_columns.py input.txt [output.txt]")
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(infile)[0] + "_trimmed.txt"

with open(infile, "r") as fin, open(outfile, "w") as fout:
    for line in fin:
        # Skip empty or whitespace-only lines
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 2:
            fout.write(f"{parts[0]}\t{parts[1]}\n")

print(f"Saved trimmed file: {outfile}")
