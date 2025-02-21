import os
import csv
import json
import random

# Define paths
input_folder = "all_flowers20_025"
output_folder1 = "all_flowers20_025_first_10"
output_folder2 = "all_flowers20_025_second_10"

# Create output directories if they don't exist
os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)

# Get list of all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# Process each CSV file
for file in csv_files:
    input_file_path = os.path.join(input_folder, file)

    # Read CSV file using standard CSV reader
    with open(input_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        data_rows = [json.loads(row[0]) for row in reader]  # Convert JSON strings to Python lists

    # Ensure the file has exactly 20 rows
    if len(data_rows) != 20:
        print(f"Skipping {file} - does not have exactly 20 rows (Detected {len(data_rows)} rows)")
        continue

    # Randomly shuffle and split data
    random.shuffle(data_rows)  # Shuffle rows randomly
    first_10 = data_rows[:10]  # Random first 10 rows
    second_10 = data_rows[10:]  # Remaining 10 rows

    # Define output file paths
    first_output_path = os.path.join(output_folder1, file)
    second_output_path = os.path.join(output_folder2, file)

    # Save first 10 rows
    with open(first_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in first_10:
            writer.writerow([json.dumps(row)])  # Convert list back to JSON string before saving

    # Save second 10 rows
    with open(second_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in second_10:
            writer.writerow([json.dumps(row)])  # Convert list back to JSON string before saving

    print(f"Processed and saved: {file}")

print("✅ Splitting complete. Check 'all_flowers20_025_first_10' and 'all_flowers20_025_second_10' folders.")