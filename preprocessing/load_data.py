
import pandas as pd

file_path = "data/Scats Data October 2006.xls"

# Load the actual traffic data
data = pd.read_excel(
    file_path,
    sheet_name="Data",
    header=1
)

# Convert Date column into datetime format
data["Date"] = pd.to_datetime(data["Date"])

# Check unique SCATS sites
print("Number of SCATS sites:")
print(data["SCATS Number"].nunique())

print("\nSCATS site numbers:")
print(sorted(data["SCATS Number"].unique()))

# Check date range
print("\nDate range:")
print(data["Date"].min(), "to", data["Date"].max())

# Check records per SCATS site
print("\nRecords per SCATS site:")
print(data.groupby("SCATS Number").size())

# Check duplicate records
print("\nDuplicate rows:")
print(data.duplicated().sum())

# Check traffic data summary
traffic_columns = [
    f"V{i:02d}" for i in range(96)
]

print("\nTraffic data summary:")
print(data[traffic_columns].describe())
