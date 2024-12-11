import numpy as np
import pandas as pd
import json
import cv2
import os
import torch
from scipy.spatial import Delaunay, ConvexHull

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import Instances, Boxes

import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer

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
CW_DISTANCE_THRESHOLD = config['cw_distance_threshold']


# Core Functions
def setup_predictor(model_file_path, num_classes, threshold=0.5):
    '''
    Setup the predictor

    Args:
        model_file_path (str): Path to the model file
    '''
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = model_file_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = 'cuda'
    return DefaultPredictor(cfg)


def fit_polygon_delaunay(contour):
    '''
    Fit a polygon using Delaunay triangulation.

    Parameters:
        contour (numpy.ndarray): Contour points as a 2D array (N, 2).

    Returns:
        numpy.ndarray: Array of vertices of the fitted polygon.
    '''
    # Ensure contour is 2D
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError('Contour must be a 2D array with shape (N, 2).')

    # Delaunay triangulation
    tri = Delaunay(contour)

    # Extract the convex hull (outermost polygon)
    hull_indices = cv2.convexHull(contour, returnPoints=False).flatten()
    hull_points = contour[hull_indices]

    return hull_points


def measure_crosswalk_dimensions(polygon):
    '''
    Measure the length and width of a crosswalk from a polygon.

    Parameters:
        polygon (numpy.ndarray): Nx2 array of polygon vertices.

    Returns:
        tuple: (length, width)
            length (float): Longest distance between two vertices (distance pedestrians walk).
            width (float): Shortest distance across the polygon (space available for pedestrians).
    '''
    # Step 1: Simplify the polygon using a convex hull
    hull = ConvexHull(polygon)
    hull_points = polygon[hull.vertices]

    # Step 2: Calculate all pairwise distances between hull points
    num_points = len(hull_points)
    pairwise_distances = []
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(hull_points[i] - hull_points[j])
            pairwise_distances.append((dist, i, j))

    # Step 3: Identify the longest distance (length)
    pairwise_distances = sorted(pairwise_distances, key=lambda x: -x[0])  # Sort by descending distance
    length, idx1, idx2 = pairwise_distances[0]  # Longest distance and its endpoints
    longest_axis = (hull_points[idx1], hull_points[idx2])

    # Step 4: Calculate perpendicular distances to the longest axis (width)
    def point_to_line_distance(point, line_start, line_end):
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)
        projection = np.dot(point_vec, line_vec) / line_len
        proj_point = line_start + projection * (line_vec / line_len)
        return np.linalg.norm(point - proj_point)

    # Measure perpendicular distances from all hull points to the longest axis
    line_start, line_end = longest_axis
    perpendicular_distances = [
        point_to_line_distance(pt, line_start, line_end) for pt in hull_points if not np.array_equal(pt, line_start) and not np.array_equal(pt, line_end)
    ]
    width = max(perpendicular_distances) if perpendicular_distances else 0  # Take the largest perpendicular distance as width

    return length, width


def measure_intersection_sides(polygon):
    '''
    Measure the lengths of the four sides of an intersection polygon by approximating it
    to a quadrilateral, with improved robustness.

    Parameters:
        polygon (numpy.ndarray): Nx2 array of vertices representing the polygon.

    Returns:
        tuple: (results, quad_points)
            results: Dictionary mapping directions to side lengths:
                     {'north': length, 'south': length, 'east': length, 'west': length}.
            quad_points: Nx2 array of quadrilateral vertices.
    '''
    # Ensure the polygon is in the correct format for OpenCV functions
    if polygon.ndim == 2:
        contour = polygon.reshape(-1, 1, 2).astype(np.int32)  # Convert to (N, 1, 2)
    else:
        contour = polygon.astype(np.int32)

    # Attempt to approximate the contour to a quadrilateral
    peri = cv2.arcLength(contour, True)
    approx = None

    # Try multiple epsilon values to find an approximation with 4 points
    for epsilon_factor in np.linspace(0.01, 0.1, 20):
        epsilon = epsilon_factor * peri
        approx_candidate = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx_candidate) == 4:
            approx = approx_candidate
            break  # Exit the loop if we find a suitable approximation

    if approx is None:
        # If we cannot approximate to a quadrilateral, use minAreaRect as a fallback
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        quad_points = np.intp(box)
    else:
        # Extract the points of the quadrilateral
        quad_points = approx[:, 0, :]  # Shape (4, 2)

    # Compute centroid of the quadrilateral
    centroid = np.mean(quad_points, axis=0)

    # Initialize directions
    directions = {'north': [], 'south': [], 'east': [], 'west': []}

    # Iterate over each side of the quadrilateral
    for i in range(4):
        pt1 = quad_points[i]
        pt2 = quad_points[(i + 1) % 4]
        length = np.linalg.norm(pt2 - pt1)

        # Compute midpoint of the side
        midpoint = (pt1 + pt2) / 2
        delta = midpoint - centroid
        angle = np.arctan2(delta[1], delta[0])  # Angle in radians

        # Assign side to a direction based on its angle
        angle_deg = np.degrees(angle) % 360  # Convert angle to degrees and normalize
        if 45 <= angle_deg < 135:  # North
            directions['south'].append({'length': length, 'points': (tuple(pt1), tuple(pt2))})
        elif 135 <= angle_deg < 225:  # West
            directions['west'].append({'length': length, 'points': (tuple(pt1), tuple(pt2))})
        elif 225 <= angle_deg < 315:  # South
            directions['north'].append({'length': length, 'points': (tuple(pt1), tuple(pt2))})
        else:  # East
            directions['east'].append({'length': length, 'points': (tuple(pt1), tuple(pt2))})

    # Summarize results by selecting the side with maximum length for each direction
    results = {}
    for dir, sides in directions.items():
        if sides:
            # Choose the side with maximum length
            max_side = max(sides, key=lambda x: x['length'])
            results[dir] = max_side
        else:
            # If no side was assigned to this direction
            results[dir] = {'length': None, 'points': None}

    return results, quad_points


def assign_crosswalks_to_directions(intersection_polygon, crosswalk_polygons):
    '''
    Assign each crosswalk to a cardinal direction relative to the intersection.

    Parameters:
        intersection_polygon (numpy.ndarray): Nx2 array of vertices for the intersection polygon.
        crosswalk_polygons (list of numpy.ndarray): List of Nx2 arrays representing crosswalk polygons.

    Returns:
        dict: Dictionary mapping directions to crosswalk polygons:
              {'north': [polygon1, polygon2, ...], 'south': [...], 'east': [...], 'west': [...]}.
    '''
    # Step 1: Compute the centroid of the intersection
    intersection_centroid = np.mean(intersection_polygon, axis=0)

    # Step 2: Initialize direction mapping
    direction_mapping = {'north': [], 'south': [], 'east': [], 'west': []}

    # Step 3: Assign each crosswalk to a direction
    for crosswalk_polygon in crosswalk_polygons:
        # Compute the centroid of the crosswalk
        crosswalk_centroid = np.mean(crosswalk_polygon, axis=0)

        # Calculate the angle of the crosswalk centroid relative to the intersection centroid
        angle = np.arctan2(crosswalk_centroid[1] - intersection_centroid[1],
                           crosswalk_centroid[0] - intersection_centroid[0])

        # Assign direction based on the angle
        if -np.pi / 4 <= angle < np.pi / 4:  # East
            direction_mapping['east'].append(crosswalk_polygon)
        elif -3 * np.pi / 4 <= angle < -np.pi / 4:  # North
            direction_mapping['north'].append(crosswalk_polygon)
        elif np.pi / 4 <= angle < 3 * np.pi / 4:  # South
            direction_mapping['south'].append(crosswalk_polygon)
        else:  # West
            direction_mapping['west'].append(crosswalk_polygon)

    return direction_mapping


def extract_features(image_file_path, predictions, class_labels, cw_dist_thrshold=20, other_dist_thrshold=100):
    '''
    Extract features from predictions, including crosswalk assignments, orientations, and intersection dimensions.
    '''
    # Load image and make predictions
    image = cv2.imread(image_file_path)
    image_dims = image.shape[:2]  # (height, width)
    # predictions = predictor(image)

    # Setup
    features = {f'{k}_total_count': 0 for k in class_labels}
    features.update({f'{k}_main_count': 0 for k in class_labels if k != 'intersection'})
    image_center = (image_dims[1] // 2, image_dims[0] // 2)  # (x_center, y_center)
    total_area = image_dims[0] * image_dims[1]

    # Extract masks and classes
    instances = predictions['instances'].to('cpu')
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()

    # # Smooth masks
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # for i, mask in enumerate(masks):
    #     mask = (mask * 255).astype(np.uint8)
    #     morphed_mask = cv2.dilate(mask, kernel, iterations=1)
    #     morphed_mask = cv2.morphologyEx(morphed_mask, cv2.MORPH_CLOSE, kernel)
    #     masks[i] = morphed_mask

    # Step 1: Check for intersections
    for i, cls in enumerate(classes):
        features[f'{class_labels[cls]}_total_count'] += 1

    # Check if there are any intersections
    if features['intersection_total_count'] == 0:
        return features

    # Step 2: Extract intersection contours and find the main intersection
    intersection_contours = []
    main_intersection = None
    min_distance_to_center = float('inf')
    # Find main intersection instance
    for i, cls in enumerate(classes):
        if class_labels[cls] == 'intersection':
            # Extract mask
            mask = masks[i]
            # Find contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                intersection_contours.append(contour)
                # Compute distance to image center
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                distance_to_center = np.linalg.norm(np.array([cx, cy]) - np.array(image_center))
                # Assign to main intersection if it's the closest to the center
                if distance_to_center < min_distance_to_center:
                    min_distance_to_center = distance_to_center
                    main_intersection = {
                        'mask': mask,
                        'contour': contour,
                        'center': (cx, cy),
                        'box': cv2.boxPoints(cv2.minAreaRect(contour))  # Corners of the bounding box
                    }
    
    # Step 3: Add main intersection dimensions to features dict
    if main_intersection:
        contour = main_intersection['contour']
        # Ensure contour is a 2D array of (x, y) points
        if contour.ndim == 3:
                contour = contour[:, 0, :]  # Convert from (N, 1, 2) to (N, 2)
        # Fit polygon, measure sides and assign directions
        polygon = fit_polygon_delaunay(contour)
        main_intersection['polygon'] = polygon
        side_lengths, _ = measure_intersection_sides(polygon)
        for direction, info in side_lengths.items():
            features[f'int_{direction}_side_length'] = info['length']

    # Step 4: Identify crosswalks at main intersection and measure dimensions
    crosswalk_counter = 0
    crosswalk_polygons = {}
    crosswalk_lengths = {}
    crosswalk_widths = {}
    for i, cls in enumerate(classes):
        if class_labels[cls] == 'crosswalk':
            # Extract mask
            mask = masks[i].astype(np.uint8)
            # Find the largest contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            # Ensure contour is a 2D array of (x, y) points
            if contour.ndim == 3:
                contour = contour[:, 0, :]  # Convert from (N, 1, 2) to (N, 2)
            # Check if the crosswalk is close to the main intersection
            close_to_intersection = False
            for point in contour:
                point_tuple = (int(point[0]), int(point[1]))
                distance = cv2.pointPolygonTest(main_intersection['contour'], point_tuple, True)
                if abs(distance) <= cw_dist_thrshold:  # Check if within certain number of pixels
                    close_to_intersection = True
                    break  # No need to check further
            # If close to intersection, add to main_int_crosswalks
            if close_to_intersection:
                polygon = fit_polygon_delaunay(contour)
                length, width = measure_crosswalk_dimensions(polygon)
                crosswalk_polygons[crosswalk_counter] = polygon
                crosswalk_lengths[crosswalk_counter] = length
                crosswalk_widths[crosswalk_counter] = width
                crosswalk_counter += 1
                features['crosswalk_main_count'] += 1
        
        cw_direction = assign_crosswalks_to_directions(main_intersection['polygon'], crosswalk_polygons.values())
        for i, direction in zip(crosswalk_polygons.keys(), cw_direction):
            features[f'cw_{direction}_side_length'] = crosswalk_lengths[i]
            features[f'cw_{direction}_side_width'] = crosswalk_widths[i]

    # Step 5: Count other road features
    for i, cls in enumerate(classes):
        if class_labels[cls] not in ['crosswalk', 'intersection']:
            # Extract mask
            mask = masks[i].astype(np.uint8)
            if not np.any(mask):
                continue
            # Find the largest contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                unified_mask = np.zeros_like(mask)
                cv2.drawContours(unified_mask, contours, -1, 255, thickness=cv2.FILLED)
                contours, _ = cv2.findContours(unified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            # Ensure contour is a 2D array of (x, y) points
            if contour.ndim == 3:
                contour = contour[:, 0, :]  # Convert from (N, 1, 2) to (N, 2)
            # Check if the crosswalk is close to the main intersection
            close_to_intersection = False
            for point in contour:
                point_tuple = (int(point[0]), int(point[1]))
                distance = cv2.pointPolygonTest(main_intersection['contour'], point_tuple, True)
                if abs(distance) <= other_dist_thrshold:  # Check if within certain number of pixels
                    close_to_intersection = True
                    crosswalk_counter += 1
                    break  # No need to check further
            # If close to intersection, add to main_int_crosswalks
            if close_to_intersection:
                features[f'{class_labels[cls]}_main_count'] += 1

    # Step 6: Calculate overhead features
    areas = {'overhead': 0, 'overpass': 0}
    for i, cls in enumerate(classes):
        if class_labels[cls] in ['overhead', 'overpass']:
            mask = masks[i].astype(np.uint8)
            areas[f'{class_labels[cls]}'] += np.sum(mask)
            if mask[image_center[1], image_center[0]]:
                features[f'{class_labels[cls]}_center'] = 1
    for key, value in areas.items():
        features[f'{key}_area'] = value / total_area

    return features


def annotations_to_model_format(dataset_dict):
    '''
    Convert annotations to mimic Detectron2 model output format.
    '''

    def polygons_to_mask(polygons, image_shape):
        '''
        Convert COCO-style polygons to a binary mask.
        '''
        mask = np.zeros(image_shape, dtype=np.uint8)
        for polygon in polygons:
            points = np.array(polygon, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [points], 1)
        return mask
    
    # Get image dims
    height = dataset_dict['height']
    width = dataset_dict['width']
    image_shape = (height, width)
    
    # Initialize arrays for masks, classes, and bounding boxes
    pred_masks = []
    pred_classes = []
    pred_boxes = []
    
    for anno in dataset_dict['annotations']:
        # Convert polygons or other masks into binary masks
        if 'segmentation' in anno:  # COCO-style segmentation
            mask = polygons_to_mask(anno['segmentation'], image_shape)
            pred_masks.append(mask)
        
        # Add class ID (map from label name to ID)
        if 'category_id' in anno:
            pred_classes.append(anno['category_id'])
        
        # Add bounding box
        if 'bbox' in anno:  # COCO-style bounding box
            x, y, width, height = anno['bbox']
            pred_boxes.append([x, y, x + width, y + height])
    
    # Convert data to torch tensors for compatibility with Detectron2
    pred_masks = torch.tensor(np.array(pred_masks), dtype=torch.bool)
    pred_classes = torch.tensor(pred_classes, dtype=torch.int64)
    pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
    
    # Create an Instances object to mimic model predictions
    instances = Instances(
        image_shape,
        pred_boxes=Boxes(pred_boxes),
        pred_classes=pred_classes,
        pred_masks=pred_masks
    )
    
    return {'instances': instances}


# Visualization Functions
def visualize_model_predictions(predictor, img_file_path, metadata, plot=True):
    img = cv2.imread(img_file_path)
    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs['instances'].to('cpu'))

    if plot:
        plt.figure(figsize=(15, 10))
        plt.imshow(v.get_image())
        plt.axis('off')
        plt.show()
    else:
        return v.get_image()


def visualize_crosswalk_measurements(predictor, img_file_path, metadata, class_labels, fit_function, plot=True):
    '''
    Visualize crosswalk measurements (length and width) based on the fitted polygon.

    Parameters:
        predictor: Detectron2 predictor object.
        img_file_path (str): Path to the image file.
        metadata: Metadata object for the visualizer.
        class_labels: List of class labels corresponding to class indices.
        fit_function: Polygon fitting function (e.g., fit_polygon_delaunay).
    '''

    # Read the image
    img = cv2.imread(img_file_path)
    outputs = predictor(img)

    # Initialize the visualizer
    from detectron2.utils.visualizer import Visualizer
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs['instances'].to('cpu'))

    # Extract predictions
    instances = outputs['instances'].to('cpu')
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()

    # Overlay length and width measurements
    for i, cls in enumerate(classes):
        if class_labels[cls] == 'crosswalk':
            mask = masks[i].astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue  # Skip if no contours are found
            
            # Use the largest contour
            contour = max(contours, key=cv2.contourArea)

            # Ensure contour is a 2D array of (x, y) points
            if contour.ndim == 3:
                contour = contour[:, 0, :]  # Convert from (N, 1, 2) to (N, 2)

            # Fit polygon using the provided fitting function
            vertices = fit_function(contour)

            # Measure dimensions
            length, width = measure_crosswalk_dimensions(vertices)

            # Draw the fitted polygon on the image
            for j in range(len(vertices)):
                pt1 = tuple(vertices[j])
                pt2 = tuple(vertices[(j + 1) % len(vertices)])  # Wrap around to the first vertex
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)  # Green lines for polygons

            # Annotate measurements
            midpoint = tuple(np.mean(vertices, axis=0).astype(int))  # Find the polygon's center
            cv2.putText(img, f'Length: {length:.2f}', (midpoint[0] - 50, midpoint[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(img, f'Width: {width:.2f}', (midpoint[0] - 50, midpoint[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if plot:
        # Show the results
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

def visualize_intersection_sides(predictor, img_file_path, metadata, class_labels, fit_function, plot=True):
    '''
    Visualize the four sides of an intersection polygon, assigning lengths and cardinal directions.

    Parameters:
        predictor: Detectron2 predictor object.
        img_file_path (str): Path to the image file.
        metadata: Metadata object for visualizer.
        class_labels: List of class labels corresponding to class indices.
        plot (bool): Whether to display the plot. If False, returns the image array.
    '''

    # Read the image
    img = cv2.imread(img_file_path)
    outputs = predictor(img)

    # Initialize the visualizer
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs['instances'].to('cpu'))

    # Extract predictions
    instances = outputs['instances'].to('cpu')
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()

    # Process intersection polygons
    for i, cls in enumerate(classes):
        if class_labels[cls] == 'intersection':
            found_int = True
            mask = masks[i].astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue  # Skip if no contours are found

            # Use the largest contour
            contour = max(contours, key=cv2.contourArea)

            # Ensure contour is a 2D array of (x, y) points
            if contour.ndim == 3:
                contour = contour[:, 0, :]  # Convert from (N, 1, 2) to (N, 2)

            # Fit polygon using the provided fitting function
            polygon = fit_function(contour)

            # Measure sides and assign directions
            try:
                side_lengths, quad_points = measure_intersection_sides(contour)
            except ValueError as e:
                print(f'Warning: {e}')
                continue  # Skip this intersection if approximation fails

            # Draw the quadrilateral
            for j in range(len(quad_points)):
                pt1 = tuple(quad_points[j])
                pt2 = tuple(quad_points[(j + 1) % len(quad_points)])  # Wrap around
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)  # Green polygon edges

            # Annotate directions and lengths
            for direction, info in side_lengths.items():
                if info['points'] is not None and info['length'] is not None:
                    pt1, pt2 = info['points']
                    midpoint = tuple(((np.array(pt1) + np.array(pt2)) / 2).astype(int))
                    cv2.putText(img, f'{direction}: {info['length']:.2f}',
                                midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if plot:
        # Show the results
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

def visualize_crosswalk_directions(predictor, img_file_path, metadata, class_labels, fit_function, plot=True):
    '''
    Visualize crosswalk polygons and their assigned directions relative to an intersection.

    Parameters:
        predictor: Detectron2 predictor object.
        img_file_path (str): Path to the image file.
        metadata: Metadata object for visualizer.
        class_labels: List of class labels corresponding to class indices.
        fit_function: Polygon fitting function (e.g., fit_polygon_delaunay).
    '''

    # Read the image
    img = cv2.imread(img_file_path)
    outputs = predictor(img)

    # Initialize the visualizer
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs['instances'].to('cpu'))

    # Extract predictions
    instances = outputs['instances'].to('cpu')
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()

    # Identify intersection and crosswalks
    intersection_polygon = None
    crosswalk_polygons = []
    for i, cls in enumerate(classes):
        mask = masks[i].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue  # Skip if no contours are found

        contour = max(contours, key=cv2.contourArea)  # Use the largest contour
        if contour.ndim == 3:
            contour = contour[:, 0, :]  # Convert to 2D array

        if class_labels[cls] == 'intersection':
            intersection_polygon = fit_function(contour)
        elif class_labels[cls] == 'crosswalk':
            crosswalk_polygons.append(fit_function(contour))

    # Assign crosswalks to directions
    if intersection_polygon is not None:

        direction_mapping = assign_crosswalks_to_directions(intersection_polygon, crosswalk_polygons)

        # Visualize intersection polygon
        for i in range(len(intersection_polygon)):
            pt1 = tuple(intersection_polygon[i])
            pt2 = tuple(intersection_polygon[(i + 1) % len(intersection_polygon)])  # Wrap around
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)  # Green for intersection polygon

        # Visualize crosswalks and annotate directions
        for direction, polygons in direction_mapping.items():
            for polygon in polygons:
                for j in range(len(polygon)):
                    pt1 = tuple(polygon[j])
                    pt2 = tuple(polygon[(j + 1) % len(polygon)])
                    cv2.line(img, pt1, pt2, (255, 0, 0), 2)  # Blue for crosswalks

                # Annotate direction at the centroid
                centroid = tuple(np.mean(polygon, axis=0).astype(int))
                cv2.putText(img, direction, centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    if plot:
        # Show the results
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

def visualize_all_in_grid(img_file_path, predictor, metadata, dataset_dicts, class_labels, fit_function):
    '''
    Visualize the outputs of four model visualization functions in a 2x2 grid.

    Parameters:
        img_file_path (str): Path to the input image file.
        predictor: Detectron2 predictor object.
        metadata: Metadata for visualizer.
        class_labels: List of class labels corresponding to class indices.
        fit_function: Polygon fitting function (e.g., fit_polygon_delaunay).
    '''
    import matplotlib.pyplot as plt

    # Generate images from each visualize function
    if dataset_dicts:
        img_annotations = visualize_annotations(img_file_path, metadata, dataset_dicts, plot=False)
    else:
        img = cv2.imread(img_file_path)
        img_annotations = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_predictions = visualize_model_predictions(predictor, img_file_path, metadata, plot=False)
    img_intersection = visualize_intersection_sides(predictor, img_file_path, metadata, class_labels, fit_function, plot=False)
    # img_crosswalk_directions = visualize_crosswalk_directions(predictor, img_file_path, metadata, class_labels, fit_function, plot=False)
    img_crosswalk_measurements = visualize_crosswalk_measurements(predictor, img_file_path, metadata, class_labels, fit_function, plot=False)

    # Create a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    # Add images to the grid
    axs[0, 0].imshow(img_annotations) # img_predictions
    axs[0, 0].set_title('Model Predictions')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(img_predictions) # img_intersection
    axs[0, 1].set_title('Intersection Sides')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(img_intersection) # img_crosswalk_directions
    axs[1, 0].set_title('Crosswalk Directions')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(img_crosswalk_measurements)
    axs[1, 1].set_title('Crosswalk Measurements')
    axs[1, 1].axis('off')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def visualize_annotations(image_file_path, metadata, dataset_dicts, plot=True):
    '''
    Visualize annotations using Visualizer's draw_dataset_dict for better annotation handling.

    :param image_file_path: Path to the image file.
    :param metadata: Metadata object containing class information.
    :param plot: Boolean flag; if True, displays the plot. If False, returns the annotated image.
    :return: Annotated image (if plot=False).
    '''
    import os

    # Determine the corresponding JSON annotation file
    annotation_path = image_file_path.replace('images', 'annotations').replace('.png', '.json')

    # Load the image
    img = cv2.imread(image_file_path)

    # Load the annotations
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    # Prepare the dataset dictionary
    dataset_dict = [
        d for d in dataset_dicts if os.path.basename(d['file_name']) == os.path.basename(image_file_path)
    ][0]

    # Create a Visualizer
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
    v = v.draw_dataset_dict(dataset_dict)

    # Display or return the image
    if plot:
        plt.figure(figsize=(15, 10))
        plt.imshow(v.get_image())
        plt.axis('off')
        plt.show()
    else:
        return v.get_image()
    

def visualize_annotations_predictions_grid(image_file_path, predictor, metadata, dataset_dicts, plot=True):
    '''
    Visualize annotations and predictions for an image in a 2x2 grid.

    :param image_file_path: Path to the image file.
    :param predictor: Detectron2 predictor object.
    :param metadata: Metadata object containing category and color mapping.
    :param plot: Boolean flag; if True, displays the plot. If False, returns the annotated image.
    :return: Annotated image (if plot=False).
    '''
    import matplotlib.pyplot as plt

    # Generate visualizations
    img_annotations = visualize_annotations(image_file_path, metadata, dataset_dicts, plot=False)
    img_predictions = visualize_model_predictions(predictor, image_file_path, metadata, plot=False)

    # Create a 2x2 grid
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    # Add images to the grid
    axs[0].imshow(img_annotations)
    axs[0].set_title('Annotations')
    axs[0].axis('off')

    axs[1].imshow(img_predictions)
    axs[1].set_title('Predictions')
    axs[1].axis('off')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# MAIN
if __name__ == '__main__':

    # Load COCO annotations for classes
    annotations_file = os.path.join(ANNOTATIONS_DIR, 'dataset.json')
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    class_labels = [x['name'] for x in coco_data['categories']]
    num_classes = len(class_labels)

    # Setup model
    model_file_path = os.path.join(MODEL_DIR, MODEL_NAME)
    predictor = setup_predictor(model_file_path, num_classes, threshold=THRESHOLD)
    """ REPLACE IN FUTURE """
    image_files = [x for x in os.listdir(IMAGES_DIR) if x.replace('.png', '.json') in os.listdir(ANNOTATIONS_DIR)] # os.listdir(IMAGES_DIR)
    """ REPLACE IN FUTURE """
    if NUM_IMAGES_TO_PROCESS:
        image_files = image_files[:NUM_IMAGES_TO_PROCESS]

    print(f'Processing {len(image_files)} images...')
    
    # Extract features
    all_features = []
    for i, image_file in enumerate(image_files):
        image_file_path = os.path.join(IMAGES_DIR, image_file)
        image = cv2.imread(image_file_path)
        predictions = predictor(image)
        features = extract_features(image_file_path, predictions, class_labels)
        features['image_name'] = image_file.replace('.png', '')
        all_features.append(features)

        if i % 50 == 0 and i > 0:
            print(f'Processed {i} images...')

    print(f'Finished processing {len(image_files)} images.')
    
    # Save results to a DataFrame
    df = pd.DataFrame(all_features)
    output_file = os.path.join(OUTPUT_DIR, 'image_features.csv')
    df.to_csv(output_file, index=False)