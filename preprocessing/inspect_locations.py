import pandas as pd

file_path = "data/Traffic_Count_Locations_with_LONG_LAT.csv"

locations = pd.read_csv(file_path)

print("Dataset shape:")
print(locations.shape)

print("\nColumn names:")
print(locations.columns.tolist())

print("\nFirst 10 rows:")
print(locations.head(10).to_string(index=False))

print("\nMissing values:")
print(locations.isnull().sum())

print("\nData types:")
print(locations.dtypes)