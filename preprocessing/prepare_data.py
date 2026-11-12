
import pandas as pd

# Load the prepared dataset
data = pd.read_csv("data/traffic_timeseries.csv")

data["Timestamp"] = pd.to_datetime(data["Timestamp"])

# Combine traffic volumes from all directions
combined_data = data.groupby(
    ["SCATS Number", "Timestamp"],
    as_index=False
).agg({
    "Traffic Volume": "sum"
})

# Sort the data
combined_data = combined_data.sort_values(
    ["SCATS Number", "Timestamp"]
)

# Check SCATS site 2000
site_data = combined_data[
    combined_data["SCATS Number"] == 2000
].copy()

print("Combined dataset shape:", combined_data.shape)

print("\nSite 2000 dataset shape:", site_data.shape)

print("\nFirst 10 readings:")
print(site_data.head(10))

print("\nMissing values:")
print(site_data.isnull().sum())

print("\nDuplicate timestamps:",
      site_data["Timestamp"].duplicated().sum())

# Save combined dataset
combined_data.to_csv(
    "data/combined_traffic_timeseries.csv",
    index=False
)

print("\nCombined dataset saved successfully!")


# Check missing 15-minute intervals for site 2000
expected_timestamps = pd.date_range(
    start=site_data["Timestamp"].min(),
    end=site_data["Timestamp"].max(),
    freq="15min"
)

missing_timestamps = expected_timestamps.difference(
    site_data["Timestamp"]
)

print("\nExpected readings:", len(expected_timestamps))
print("Actual readings:", len(site_data))
print("Missing timestamps:", len(missing_timestamps))

print("\nFirst missing timestamps:")
print(missing_timestamps[:10])

# Count readings for each day
site_data["Date"] = site_data["Timestamp"].dt.date

daily_counts = site_data.groupby("Date").size()

print("\nDaily reading counts:")
print(daily_counts.to_string())

print("\nDays with incomplete readings:")
print(daily_counts[daily_counts != 96])