import json
import os
import requests
import pickle
from multiprocessing import Pool
from time import time
from datetime import datetime
from config import config

def get_511_camera_img(camera_response, file_path, in_scope_ids):
    if not camera_response['Disabled'] and camera_response['ID'] in in_scope_ids:
        url = camera_response['Url']
        id = camera_response['ID']
        name = camera_response['Name']
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'{file_path}/{id}.jpg', 'wb') as file:
                file.write(response.content)
            return {id: name}

    return None

if __name__ == '__main__':

    PRINT_QUERY_RESULTS = config['PRINT_QUERY_RESULTS']
    start = time()

    with open('data/api_keys.json', 'r') as file:
        keys = json.load(file)
    api_key = keys['511']
    response_format = 'json'

    url = f'https://511ny.org/api/getcameras?key={api_key}&format={response_format}'

    with open('data/nyc_camera_ids.pkl', 'rb') as f:
        nyc_camera_ids = pickle.load(f)

    # Make the HTTP GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Process the response if needed. This example assumes a JSON response.
        cameras_dict = response.json()

    time_stamp = datetime.now().strftime('%m.%d_%H.%M.%S')
    file_path = f'data/raw/511_imgs/{time_stamp}/'
    os.makedirs(file_path, exist_ok=True)

    args_list = [(camera, file_path, nyc_camera_ids) for camera in cameras_dict]

    with Pool(processes=15) as pool:
        results = pool.starmap(get_511_camera_img, args_list)

    # Filter out None values and combine dictionaries
    camera_names = {k: v for result in results if result for k, v in result.items()}

    with open(f'{file_path}/camera_names.json', 'w') as f:
        json.dump(camera_names, f)

    if PRINT_QUERY_RESULTS: 
        print(f'{len(camera_names)} of {len(cameras_dict)} cameras processed in {(time() - start) / 60:.2f} minutes')