import pandas as pd

# Load the original traffic dataset
file_path = "data/Scats Data October 2006.xls"

data = pd.read_excel(
    file_path,
    sheet_name="Data",
    header=1
)

# Identify the 96 traffic columns
traffic_columns = [f"V{i:02d}" for i in range(96)]

# Convert the wide dataset into long format
long_data = data.melt(
    id_vars=[
        "SCATS Number",
        "Location",
        "NB_LATITUDE",
        "NB_LONGITUDE",
        "Date"
    ],
    value_vars=traffic_columns,
    var_name="Interval",
    value_name="Traffic Volume"
)

# Extract the calendar date
long_data["Date"] = pd.to_datetime(long_data["Date"]).dt.normalize()

# Convert V00, V01 ... V95 into 15-minute intervals
interval_number = long_data["Interval"].str[1:].astype(int)

long_data["Time"] = pd.to_timedelta(
    interval_number * 15,
    unit="minutes"
)

# Create the timestamp for each traffic reading
long_data["Timestamp"] = long_data["Date"] + long_data["Time"]

# Sort the data
long_data = long_data.sort_values(
    ["SCATS Number", "Timestamp"]
).reset_index(drop=True)

# Save the processed dataset
output_path = "data/traffic_timeseries.csv"
long_data.to_csv(output_path, index=False)

# Display results
print("Reshaping completed!")
print("\nDataset shape:")
print(long_data.shape)

print("\nFirst 10 records:")
print(long_data.head(10).to_string(index=False))

print("\nMissing values:")
print(long_data.isnull().sum())

print("\nSaved to:", output_path)
