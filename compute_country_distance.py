# compute_country_distances.py
import polars as pl
import numpy as np
from itertools import combinations
import os

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers

    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except (ValueError, TypeError):
        return float('inf') # Return infinity if conversion fails

    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def main():
    input_csv = 'country.csv'
    output_parquet = 'country_distances.parquet'
    output_dir = os.path.dirname(output_parquet)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading country data from {input_csv}...")
    try:
        country_df = pl.read_csv(input_csv)
        # Ensure necessary columns exist and are correct type
        country_df = country_df.select(
            pl.col('country').cast(pl.Utf8).alias('code'),
            pl.col('latitude').cast(pl.Float64),
            pl.col('longitude').cast(pl.Float64)
        ).drop_nulls() # Drop rows where coordinates or code are missing
        print(f"Loaded {len(country_df)} countries with valid data.")
    except Exception as e:
        print(f"Error loading or processing {input_csv}: {e}")
        return

    if country_df.is_empty():
        print("No valid country data found. Exiting.")
        return

    countries = country_df.to_dicts()
    country_lookup = {c['code']: (c['latitude'], c['longitude']) for c in countries}

    print("Calculating distances between country pairs...")
    distance_data = []
    total_pairs = len(countries) * (len(countries) - 1) // 2
    count = 0
    progress_step = max(1, total_pairs // 100) # Update progress every 1%

    for (code1, code2) in combinations(country_lookup.keys(), 2):
        lat1, lon1 = country_lookup[code1]
        lat2, lon2 = country_lookup[code2]
        dist = calculate_distance(lat1, lon1, lat2, lon2)

        # Add both directions for easier lookup later
        distance_data.append({'code_from': code1, 'code_to': code2, 'distance': dist})
        distance_data.append({'code_from': code2, 'code_to': code1, 'distance': dist})

        count += 1
        if count % progress_step == 0:
            print(f"Progress: {count / total_pairs * 100:.1f}% completed.", end='\r')

    # Add zero distance for same country
    for code in country_lookup.keys():
        distance_data.append({'code_from': code, 'code_to': code, 'distance': 0.0})

    print("\nCalculation complete.")

    print("Creating DataFrame...")
    try:
        distance_df = pl.DataFrame(distance_data, schema={
            'code_from': pl.Utf8,
            'code_to': pl.Utf8,
            'distance': pl.Float64
        })

        print(f"Saving distances to {output_parquet}...")
        distance_df.write_parquet(output_parquet, compression='zstd')
        print("Successfully saved country distances.")
    except Exception as e:
        print(f"Error creating DataFrame or saving Parquet file: {e}")

if __name__ == "__main__":
    main()