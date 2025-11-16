import pandas as pd

df = pd.read_csv("calendar.csv")
df2 = pd.read_csv("listings.csv")


prices = df["price"].dropna().astype(str)
prices2 = df2["price"].dropna().astype(str)

mask_invalid = ~prices.str.strip().str.startswith("$")
mask_invalid2 = ~prices2.str.strip().str.startswith("$")

invalid_count = mask_invalid.sum() + mask_invalid2.sum()

print(f"Number of prices not starting with '$': {invalid_count}")

if invalid_count > 0:
    print("\nExamples of invalid prices from calendar.csv:")
    print(prices[mask_invalid].head(10))

    print("\nExamples of invalid prices from listings.csv:")
    print(prices2[mask_invalid2].head(10))
else:
    print("All non-empty price values start with '$'.")

print("\nHOST LOCATIONS")

if "host_location" not in df2.columns:
    print("Column 'host_location' not found in listings.csv")
else:
    locations = df2["host_location"].dropna().astype(str).str.strip()
    location_counts = locations.value_counts()

    print("\nAvailable locations and number of listings:")
    print(location_counts)

    print("\nTotal unique locations:", len(location_counts))
