import pandas as pd
import requests
import json
import os
import time
from skimage.metrics import structural_similarity as ssim

import cv2
from PIL import Image
from io import BytesIO


"""CONFIG"""
# API keys
with open('data/api_keys.json', 'r') as file:
    keys = json.load(file)

maps_api_key = keys['Google']
geonames_username = keys['Geonames']

# Config
with open('config.json', 'r') as file:
    config = json.load(file)

REFERENCE_FILE = config['get_google_images']['starting_file']
NUM_RECORDS_TO_PROCESS = config['get_google_images']['num_records_to_process']
CHECK_FOR_ERRORS = config['get_google_images']['check_for_errors']


"""FUNCTIONS"""
# Get satellite image
def get_satellite_image(lat, lon, zoom=20, size="640x640", scale=2, map_type="satellite", api_key=maps_api_key):
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": size,
        "scale": scale,
        "maptype": map_type,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    image = Image.open(BytesIO(response.content))
    return image


# Check for errors
def check_for_errors(images_dir, template_path):
    error_images = []

    # Load the error template
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    # Loop through all images in the directory
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)

        # Read the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Skip files that can't be read
            continue
        
        # Resize the image to match the template size (if needed)
        img_resized = cv2.resize(img, (template.shape[1], template.shape[0]))

        # Compute SSIM between the template and the image
        score, _ = ssim(template, img_resized, full=True)
        if score > 0.99:  # Adjust threshold as needed
            error_images.append(image_file.replace('.png', ''))

    # Save or print the list of error images
    print(f"Found {len(error_images)} error images")

    with open('data/processed/error_images.txt', 'w') as f:
        f.write('\n'.join(error_images))

    return error_images

# Main
if __name__ == '__main__':
    # Read in intersections data
    intersections = pd.read_csv('data/reference/int_flags_deduped.csv')
    intersections = intersections.dupe_key.to_list()

    # # Get list of previously processed images
    # processed_images = os.listdir('data/images')
    # processed_images = [x.split('.png')[0] for x in processed_images if '.png' in x]

    # # Check for errors
    # if CHECK_FOR_ERRORS:
    #     error_images = check_for_errors('data/images', 'data/reference/error_image.png')
    #     processed_images = [x for x in processed_images if x not in error_images]

    # # Check for already annotated images
    # annotations_files = os.listdir('data/annotations')
    # annotated_images = [x.replace('.json', '') for x in annotations_files]

    # # Filter out already processed images
    # intersections_to_process = list(set(intersections) - set(processed_images))
    # intersections_to_process = list(set(intersections_to_process) - set(annotated_images))
    intersections_to_process = pd.read_csv('data/reference/error_images.csv').squeeze().tolist()

    print(f'Processing {len(intersections_to_process)} records out of {len(intersections)} total intersections...')

    start = time.time()
    # Get satellite images
    for idx, intersection_key in enumerate(intersections_to_process):
        lat, lng = intersection_key.split('_')
        img = get_satellite_image(lat, lng)
        img.save(f'data/images/{intersection_key}.png')
        
        if NUM_RECORDS_TO_PROCESS and idx >= NUM_RECORDS_TO_PROCESS:
            print(f"Script finished with {NUM_RECORDS_TO_PROCESS} images processed.")
            break
        
        if idx % 500 == 0:
            elapsed = time.time() - start
            print(f"Processed {idx} images...Elapsed time: {elapsed/60:.2f} minutes")

    if len(intersections_to_process):
        print(f'Script finished with {len(intersections_to_process)} images processed.')
        print(f'Elapsed time: {(time.time() - start)/60:.2f} minutes')
    else:
        print('Script finished with no images processed.')