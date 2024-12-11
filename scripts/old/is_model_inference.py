import numpy as np
import pandas as pd
import json
import cv2
import os
import re
from datetime import datetime

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Config 
with open('config.json', 'r') as f:
    config = json.load(f)['inference']

MODEL_DIR = config['model_dir']
MODEL_NAME = config['model_name']
IMAGES_DIR = config['images_dir']
ANNOTATIONS_DIR = config['annotations_dir']
NUM_IMAGES_TO_PROCESS = config['num_images_to_process']
OUTPUT_DIR = config['output_dir']
THRESHOLD = config['threshold']
POLYGON_EPS_START = config['polygon_eps_start']
POLYGON_EPS_STEP = config['polygon_eps_step']
POLYGON_MAX_ITER = config['polygon_max_iter']
CW_DISTANCE_THRESHOLD = config['cw_distance_threshold']


# Load COCO annotations for classes
annotations_file = os.path.join(ANNOTATIONS_DIR, 'dataset.json')
with open(annotations_file, 'r') as f:
    coco_data = json.load(f)
class_labels = [x['name'] for x in coco_data['categories']]
num_classes = len(class_labels)


# Functions
def get_latest_model(model_dir, prefix='model_', suffix='.pth'):
    """
    Finds the latest model file based on timestamped filenames.
    
    Args:
        model_dir (str): Path to the directory containing model files.
        prefix (str): The common prefix of model filenames.
        suffix (str): The file extension of model files (default is '.pth').
    
    Returns:
        str: The path to the latest model file, or None if no models are found.
    """
    # Use glob to find all files matching the pattern
    pattern = re.compile(r"^model_\d{2}\.\d{2}_\d{2}\.\d{2}\.pth$")
    model_files = os.listdir(model_dir)  # glob(os.path.join(model_dir, f"{prefix}*{suffix}"))
    model_files = [f for f in model_files if re.match(pattern, f)]
    if not model_files:
        return None  # No models found
    
    # Extract timestamps and sort by datetime
    model_files.sort(key=lambda x: datetime.strptime(x.split(prefix)[-1].split(suffix)[0], "%m.%d_%H.%M"), reverse=True)

    # Take latest model and return its path
    latest_model = model_files[0]
    latest_model_path = os.path.join(model_dir, latest_model)

    return latest_model_path


def setup_predictor(model_file_path, num_classes, threshold=0.5):
    """
    Setup the predictor

    Args:
        model_file_path (str): Path to the model file
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = model_file_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cuda"
    return DefaultPredictor(cfg)

def fit_four_sided_polygon(mask, epsilon_start=0.02, epsilon_step=0.005, max_iterations=20):
    """
    Fit a four-sided polygon to the contour of a segmentation mask.

    Parameters:
        mask (numpy.ndarray): Binary segmentation mask (0 for background, 255 for foreground).
        max_iterations (int): Maximum number of attempts to adjust epsilon to achieve 4 vertices.
        epsilon_start (float): Starting epsilon as a fraction of the arc length.
        epsilon_step (float): Step size for adjusting epsilon if not 4 vertices are found.

    Returns:
        tuple: (vertices, side_lengths)
            vertices (numpy.ndarray): Array of four vertices of the polygon.
            side_lengths (list): Lengths of the sides of the polygon.
    """
    # Ensure the mask is uint8
    if mask.dtype == np.bool_:
        mask = (mask * 255).astype(np.uint8)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the mask.")

    # Choose the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Iteratively adjust epsilon to fit a four-sided polygon
    epsilon = epsilon_start * cv2.arcLength(contour, True)
    for _ in range(max_iterations):
        # Approximate the contour with a polygon
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            # Successfully approximated as a quadrilateral
            vertices = approx.reshape(-1, 2)
            
            # Calculate side lengths
            side_lengths = []
            for i in range(4):
                pt1 = vertices[i]
                pt2 = vertices[(i + 1) % 4]  # Ensure the last point connects to the first
                length = np.linalg.norm(pt2 - pt1)
                side_lengths.append(length)
            
            return vertices, side_lengths
        
        # Adjust epsilon for the next iteration
        epsilon += epsilon_step * cv2.arcLength(contour, True)

    # If unable to approximate as a quadrilateral after max_iterations
    raise ValueError("Failed to approximate a four-sided polygon within the maximum iterations.")


def extract_features(
        predictions, class_labels, image_dims, 
        epsilon=0.02, epsilon_step=0.005, max_iterations=20, cw_dist_thrshold=20
    ):
    """
    Extract features from predictions, including crosswalk assignments, orientations, and intersection dimensions.
    """
    # Setup
    features = {
        "intersection_count": 0,
        "crosswalk_count": 0,
        "bikeLane_count": 0,
        "parkingLane_count": 0,
        "doubleYellow_count": 0,
        "median_count": 0,
        "overhead_count": 0,
        "int_north_side_length": 0,
        "int_south_side_length": 0,
        "int_east_side_length": 0,
        "int_west_side_length": 0,
        # "cw_north_side_length": 0,
        # "cw_north_side_width": 0,
        # "cw_south_side_length": 0,
        # "cw_south_side_width": 0,
        # "cw_east_side_length": 0,
        # "cw_east_side_width": 0,
        # "cw_west_side_length": 0,
        # "cw_west_side_width": 0
        "cw1_length": 0,
        "cw1_width": 0,
        "cw2_length": 0,
        "cw2_width": 0,
        "cw3_length": 0,
        "cw3_width": 0,
        "cw4_length": 0,
        "cw4_width": 0
    }
    image_center = (image_dims[1] // 2, image_dims[0] // 2)  # (x_center, y_center)

    # Extract masks and classes
    instances = predictions["instances"].to("cpu")
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()
    # boxes = instances.pred_boxes.tensor.numpy()

    # Step 1: Extract feature counts
    for i, cls in enumerate(classes):
        # Update feature counts
        features[f'{class_labels[cls]}_count'] += 1

    # Step 2: Extract intersection contours and find the main intersection
    # intersection_contours = []
    main_intersection = None
    min_distance_to_center = float("inf")
    # Find intersection instances
    for i, cls in enumerate(classes):
        if class_labels[cls] == "intersection":
            # Find contours
            mask = masks[i]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = contours[0]
                # intersection_contours.append(contour)
                # Compute distance to image center
                M = cv2.moments(contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                distance_to_center = np.linalg.norm(np.array([cx, cy]) - np.array(image_center))
                # Assign to main intersection if it's the closest to the center
                if distance_to_center < min_distance_to_center:
                    min_distance_to_center = distance_to_center
                    main_intersection = {
                        "mask": mask,
                        "contour": contour,
                        "center": (cx, cy),
                        "box": cv2.boxPoints(cv2.minAreaRect(contour))  # Corners of the bounding box
                    }
    
    # Step 3: Add main intersection dimensions to features dict
    if main_intersection:
        # Calculate lengths of the main intersection's sides
        contour = main_intersection["contour"]
        simplified_contour = cv2.approxPolyDP(contours[0], epsilon * cv2.arcLength(contours[0], True), True)
        if len(simplified_contour) == 4:
            # Assign lengths to north, south, east, west
            # lengths = compute_lengths_and_widths(simplified_contour)
            _, lengths = fit_four_sided_polygon(
                main_intersection["mask"], epsilon_start=epsilon, epsilon_step=epsilon_step, max_iterations=max_iterations
            )
            features["int_north_side_length"] = lengths[0]
            features["int_south_side_length"] = lengths[1]
            features["int_east_side_length"] = lengths[2]
            features["int_west_side_length"] = lengths[3]

    # Step 4: Identify crosswalks at main intersection and measure dimensions
    crosswalk_counter = 0
    for i, cls in enumerate(classes):
        if class_labels[cls] == "crosswalk":
            # Find contours
            mask = masks[i].astype(np.uint8)
            # mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            crosswalk_contour = max(contours, key=cv2.contourArea)  # Use the largest contour
            _, lengths = fit_four_sided_polygon(
                mask, epsilon_start=epsilon, epsilon_step=epsilon_step, max_iterations=max_iterations
            )
            if len(lengths) == 4:
                # Check proximity to intersection
                close_to_intersection = False
                # Check if any point in the crosswalk is within 10 pixels of the intersection
                for point in crosswalk_contour:
                    point_tuple = (int(point[0][0]), int(point[0][1])) # Extract (x, y) from the contour point
                    int_contour = main_intersection["contour"]
                    distance = cv2.pointPolygonTest(int_contour, point_tuple, True)
                    if abs(distance) <= cw_dist_thrshold:  # Check if within certain number of pixels
                        close_to_intersection = True
                        crosswalk_counter += 1
                        break  # No need to check further
                # If close to intersection, update crosswalk dimensions
                if close_to_intersection:
                    dim1 = np.max([lengths[0], lengths[2]])
                    dim2 = np.max([lengths[1], lengths[3]])
                    length = np.max([dim1, dim2])
                    width = np.min([dim1, dim2])
                    features[f'cw{crosswalk_counter}_length'] = length
                    features[f'cw{crosswalk_counter}_width'] = width
    
    return features

# MAIN
def main():
    # Setup model
    if MODEL_NAME:
        model_file_path = os.path.join(MODEL_DIR, MODEL_NAME)
    else:
        model_file_path = get_latest_model(MODEL_DIR)
    predictor = setup_predictor(model_file_path, num_classes, threshold=THRESHOLD)

    # Setup images
    image_files = os.listdir(IMAGES_DIR)
    
    ## FOR TESTING ONLY - TO BE REMOVED ##
    annotations_files = os.listdir(ANNOTATIONS_DIR)
    image_files = [f for f in image_files if f.replace('.png', '.json') in annotations_files]
    ## FOR TESTING ONLY - TO BE REMOVED ##
    
    if NUM_IMAGES_TO_PROCESS:
        image_files = image_files[:NUM_IMAGES_TO_PROCESS]
    
    # Extract features
    all_features = []
    for image_file in image_files:
        image_path = os.path.join(IMAGES_DIR, image_file)
        image = cv2.imread(image_path)
        image_dims = image.shape[:2]
        
        predictions = predictor(image)
        features = extract_features(predictions, class_labels, image_dims)
        features["image_name"] = image_file.replace('.png', '')
        all_features.append(features)

    df = pd.DataFrame(all_features)
    file_name = os.path.join(OUTPUT_DIR, 'image_features.csv')
    df.to_csv(file_name, index=False)