import pandas as pd
import requests
import time
import json
import os
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

# API KEYS
with open('data/api_keys.json', 'r') as file:
    keys = json.load(file)
geonames_username = keys['Geonames']

# CONFIG
with open('config.json', 'r') as file:
    config = json.load(file)
REFERENCE_FILE = config['rev_geocode_intersections']['starting_file']
NUM_RECORDS_TO_PROCESS = config['rev_geocode_intersections']['num_records_to_process']

# FUNCTIONS
def wait_until_next_hour():
    # Get the current time
    now = datetime.now()
    
    # Calculate the next hour
    next_hour = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
    
    # Calculate the sleep duration in seconds
    sleep_duration = (next_hour - now).total_seconds()
    
    # Pause execution
    print(f"Pausing for {sleep_duration/60:.0f} minutes until the next hour...")
    time.sleep(sleep_duration)

def check_for_intersection(coords, distance_threshold=0.02, api_key=geonames_username):
    url = f"http://api.geonames.org/findNearestIntersectionJSON?lat={coords[0]}&lng={coords[1]}&username={api_key}"
    
    while True:  # Infinite loop to retry the same record if limit is hit
        response = requests.get(url)
        if response.status_code == 200:
            # Process the response
            data = response.json()
            
            # Check for limit exceeded message
            if 'status' in data and data['status']['message'].startswith('the hourly limit'):
                print(f"Hourly limit exceeded: {data['status']['message']}")
                wait_until_next_hour()  # Wait for an hour before retrying
                continue  # Retry the same request after the wait
            
            # Check for intersection
            if 'intersection' in data:
                if float(data['intersection']['distance']) <= distance_threshold:
                    # Return true if intersection within the threshold
                    return coords[0], coords[1], True, data['intersection']
                else:
                    # Return false if intersection not within the threshold
                    return coords[0], coords[1], False, data['intersection']
        
        # Return false / None if no data found
        return coords[0], coords[1], False, None

def pool_check_for_intersection(coords):
    pool_size = cpu_count()
    with Pool(pool_size) as p:
        results = p.map(check_for_intersection, coords)
    return results

# if __name__ == "__main__":
#     # DATA
#     df = pd.read_csv(REFERENCE_FILE, low_memory=False)
#     coords = df[['LATITUDE', 'LONGITUDE']].values.tolist()

#     # Load previously processed coordinates (if any)
#     previously_processed = os.path.isfile('data/processed/intersection_flags.csv')
    
#     if previously_processed:
#         previously_processed = pd.read_csv('data/processed/intersection_flags.csv')
#         previously_processed = previously_processed[~pd.isna(previously_processed['data'])]
#         processed_coords = [tuple(coord) for coord in previously_processed[['lat', 'lng']].values.tolist()]
#         add_header = False
#     else:
#         processed_coords = []
#         add_header = True

#     # Filter out already processed coordinates
#     coords_to_process = list(set([tuple(coord) for coord in coords if tuple(coord) not in processed_coords]))
#     if IDX:
#         start, end = IDX
#         coords_to_process = coords_to_process[start:end]

#     # RUN
#     if coords_to_process:
#         results = pool_check_for_intersection(coords_to_process)
#         results_df = pd.DataFrame(results, columns=['lat', 'lng', 'intersection', 'data'])

#         # OUTPUT
#         results_df.to_csv('data/processed/intersection_flags.csv', index=False, header=add_header, mode='a')
    
#     else:
#         print("All coordinates have already been processed.")

# DATA
df = pd.read_csv(REFERENCE_FILE, low_memory=False)
coords = df[['LATITUDE', 'LONGITUDE']].values.tolist()

# Load previously processed coordinates (if any)
previously_processed = os.path.isfile('data/processed/intersection_flags.csv')

if previously_processed:
    previously_processed = pd.read_csv('data/processed/intersection_flags.csv')
    previously_processed = previously_processed[~pd.isna(previously_processed['data'])]
    processed_coords = [tuple(coord) for coord in previously_processed[['lat', 'lng']].values.tolist()]
    add_header = False
else:
    processed_coords = []
    add_header = True

# Filter out already processed coordinates
coords_to_process = list(set([tuple(coord) for coord in coords if tuple(coord) not in processed_coords]))
if NUM_RECORDS_TO_PROCESS:
    coords_to_process = coords_to_process[0:NUM_RECORDS_TO_PROCESS]

# RUN
if coords_to_process:
    results = []
    for i, coord in enumerate(coords_to_process):
        results.append(check_for_intersection(coord))
    results_df = pd.DataFrame(results, columns=['lat', 'lng', 'intersection', 'data'])

    # OUTPUT
    results_df.to_csv('data/processed/intersection_flags.csv', index=False, header=add_header, mode='a')

else:
    print("All coordinates have already been processed.")