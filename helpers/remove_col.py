# delete_third_column.py

# Input and output file paths
input_file = r"C:\Users\niket\Documents\eeg\biopac_eeg\data\Niketh_Baseline.txt"
output_file = "output.txt"

# Open input file and read lines
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Split the line into columns (whitespace or comma separated)
        parts = line.strip().split()
        
        # Check if there are at least 3 columns
        if len(parts) >= 3:
            # Remove the third column (index 2)
            del parts[2]
        
        # Write the modified line back
        outfile.write("\t".join(parts) + "\n")

print(f"Third column deleted. Result saved to '{output_file}'.")