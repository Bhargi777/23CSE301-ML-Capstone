import pandas as pd
from pathlib import Path

# Folder containing the .dat files
data_folder = Path("gas_sensor")

# List to store all batches
all_data = []

# Read batch1.dat to batch10.dat
for file in sorted(data_folder.glob("batch*.dat")):

    rows = []

    with open(file, "r") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # First value = class;concentration
            class_value, concentration = parts[0].split(";")

            # Remaining values = feature:value
            row = {
                "class": int(class_value),
                "concentration": float(concentration)
            }

            for item in parts[1:]:
                feature, value = item.split(":")
                row[f"feature_{feature}"] = float(value)

            rows.append(row)

    batch_df = pd.DataFrame(rows)
    all_data.append(batch_df)

    print(f"{file.name}: {len(batch_df)} rows")

# Combine all batches
gas_sensor = pd.concat(all_data, ignore_index=True)

# Save as CSV
output_file = data_folder / "gas_sensor.csv"
gas_sensor.to_csv(output_file, index=False)

print("\nConversion complete!")
print("Shape:", gas_sensor.shape)
print("Saved to:", output_file)