import numpy as np
import pandas as pd
import json
import cv2
import os
import torch
import time
from datetime import datetime
from scipy.spatial import Delaunay, ConvexHull
from pycocotools import mask as mask_util

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
PREDICTIONS_DIR = config['preidctions_dir']
THRESHOLD = config['threshold']
CLASS_THRESHOLDS = {int(k): v for k, v in config['class_thresholds'].items()}
CW_DISTANCE_THRESHOLD = config['cw_distance_threshold']


# Core Functions
def setup_predictor(
        model_file_path, num_classes, threshold=0.7, 
        class_thresholds={0: 0.7, 1: 0.6, 2: 0.8, 3: 0.4, 4: 0.9, 5: 0.4, 6: 0.7, 7: 0.7}
    ): 
    '''
    Setup the predictor

    Args:
        model_file_path (str): Path to the model file
    '''
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = model_file_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0 if class_thresholds else threshold
    cfg.MODEL.DEVICE = 'cuda'
    predictor = DefaultPredictor(cfg)

    # Define custom predictor with class-specific thresholds
    def custom_predictor(image):
        outputs = predictor(image)
        if class_thresholds:
            instances = outputs['instances']
            scores = instances.scores
            pred_classes = instances.pred_classes
            keep = []
            for i in range(len(scores)):
                cls = pred_classes[i].item()
                if scores[i] >= class_thresholds.get(cls, 0.0):  # Default to 0.0 if class not in thresholds
                    keep.append(i)
            outputs['instances'] = instances[torch.tensor(keep, dtype=torch.long)]
        return outputs
    
    return custom_predictor # DefaultPredictor(cfg)


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

    # Check for sufficient points for Delaunay
    if contour.shape[0] < 4:
        # If not enough points, use convex hull as fallback
        if contour.shape[0] >= 3:
            hull_indices = cv2.convexHull(contour, returnPoints=False).flatten()
            hull_points = contour[hull_indices]
            return hull_points
        else:
            # If fewer than 3 points, return the contour itself
            return contour

    # Delaunay triangulation
    tri = Delaunay(contour)

    # Extract the convex hull (outermost polygon)
    hull_indices = cv2.convexHull(contour, returnPoints=False).flatten()
    hull_points = contour[hull_indices]

    return hull_points


def measure_intersection_sides(polygon):
    '''
    Measure the lengths of the four sides of an intersection polygon by approximating it
    to a quadrilateral, with consistent indexing and deterministic vertex ordering.

    Parameters:
        polygon (numpy.ndarray): Nx2 array of vertices representing the polygon.

    Returns:
        tuple: (side_lengths, quad_points)
            side_lengths: List of side lengths, indexed in a consistent clockwise order.
            quad_points: Nx2 array of quadrilateral vertices, ordered clockwise.
    '''
    # Ensure the polygon is in the correct format for OpenCV functions
    if polygon.ndim == 2:
        contour = polygon.reshape(-1, 1, 2).astype(np.int32)  # Convert to (N, 1, 2)
    else:
        contour = polygon.astype(np.int32)

    # Approximate the contour to a quadrilateral
    peri = cv2.arcLength(contour, True)
    approx = None

    for epsilon_factor in np.linspace(0.01, 0.1, 20):
        epsilon = epsilon_factor * peri
        approx_candidate = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx_candidate) == 4:
            approx = approx_candidate
            break

    if approx is None:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        quad_points = np.intp(box)
    else:
        quad_points = approx[:, 0, :]  # Extract vertices as Nx2 array

    # Find the top-most point as the reference (smallest y; break ties with x)
    top_point_idx = np.argmin(quad_points[:, 1])
    if np.sum(quad_points[:, 1] == quad_points[top_point_idx, 1]) > 1:  # Tie in y-coordinate
        top_point_idx = np.argmin(quad_points[:, 0][quad_points[:, 1] == quad_points[top_point_idx, 1]])

    # Rotate points to start from the top-most point and go clockwise
    quad_points = np.roll(quad_points, -top_point_idx, axis=0)
    center = np.mean(quad_points, axis=0)

    # Sort the remaining points based on their angle from the center
    quad_points = sorted(quad_points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    quad_points = np.array(quad_points)

    # Measure lengths of the sides
    side_lengths = []
    for i in range(4):
        pt1 = quad_points[i]
        pt2 = quad_points[(i + 1) % 4]
        length = np.linalg.norm(np.array(pt2) - np.array(pt1))
        side_lengths.append(length)

    return side_lengths, quad_points


def assign_crosswalks_to_sides(intersection_polygon, crosswalk_polygons):
    '''
    Assign each crosswalk to the closest side of the intersection.

    Parameters:
        intersection_polygon (numpy.ndarray): Nx2 array of vertices for the intersection polygon,
                                               ordered consistently with measure_intersection_sides().
        crosswalk_polygons (list of numpy.ndarray): List of Nx2 arrays representing crosswalk polygons.

    Returns:
        dict: Dictionary mapping side indices to crosswalk polygons:
              {1: [polygon1, polygon2, ...], 2: [...], 3: [...], 4: [...]}.
    '''
    # Step 1: Define intersection sides as line segments
    sides = [(intersection_polygon[i], intersection_polygon[(i + 1) % len(intersection_polygon)])
             for i in range(len(intersection_polygon))]

    # Step 2: Initialize mapping for crosswalk assignments
    side_mapping = {i + 1: [] for i in range(len(sides))}

    # Step 3: Assign each crosswalk to the closest side
    for crosswalk_polygon in crosswalk_polygons:
        # Compute the centroid of the crosswalk
        crosswalk_centroid = np.mean(crosswalk_polygon, axis=0)

        # Find the closest side
        closest_side = None
        min_distance = float('inf')
        for i, (start, end) in enumerate(sides):
            # Calculate perpendicular distance from crosswalk centroid to the side
            distance = point_to_line_distance(crosswalk_centroid, start, end)
            if distance < min_distance:
                min_distance = distance
                closest_side = i + 1  # Index sides from 1 to 4

        # Assign the crosswalk to the closest side
        if closest_side is not None:
            side_mapping[closest_side].append(crosswalk_polygon)

    return side_mapping


def point_to_line_distance(point, line_start, line_end):
    '''
    Calculate the perpendicular distance from a point to a line segment.

    Parameters:
        point (numpy.ndarray): 1D array [x, y] of the point.
        line_start (numpy.ndarray): 1D array [x, y] of the start of the line segment.
        line_end (numpy.ndarray): 1D array [x, y] of the end of the line segment.

    Returns:
        float: Perpendicular distance from the point to the line segment.
    '''
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return np.linalg.norm(point_vec)  # Degenerate case: line is a point

    projection = np.dot(point_vec, line_vec) / line_len
    if projection < 0:
        return np.linalg.norm(point_vec)  # Closest to line_start
    elif projection > line_len:
        return np.linalg.norm(np.array(point) - np.array(line_end))  # Closest to line_end
    else:
        proj_point = np.array(line_start) + projection * (line_vec / line_len)
        return np.linalg.norm(np.array(point) - proj_point)


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
    # Check for sufficient points
    if polygon.shape[0] < 3:
        if polygon.shape[0] == 2:
            # If only two points, calculate the distance directly as length; width is 0
            length = np.linalg.norm(polygon[0] - polygon[1])
            return length, 0
        elif polygon.shape[0] == 1:
            # If only one point, length and width are 0
            return 0, 0
        else:
            raise ValueError('Polygon must have at least 1 point.')
    
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


def extract_features(image_file_path, predictions, class_labels, cw_dist_thrshold=20, other_dist_thrshold=100):
    '''
    Extract features from predictions, including crosswalk assignments, orientations, and intersection dimensions.
    '''
    # Load image and make predictions
    image = cv2.imread(image_file_path)
    image_dims = image.shape[:2]

    # Setup
    features = {f'{k}_total_count': 0 for k in class_labels}
    features.update({f'{k}_main_count': 0 for k in class_labels if k != 'intersection'})
    image_center = (image_dims[1] // 2, image_dims[0] // 2)  # (x_center, y_center)
    total_area = image_dims[0] * image_dims[1]

    # Extract masks and classes
    instances = predictions['instances'].to('cpu')
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()

    # Step 1: Calculate overhead features
    areas = {'overhead': 0, 'overpass': 0}
    for i, cls in enumerate(classes):
        if class_labels[cls] in ['overhead', 'overpass']:
            mask = masks[i].astype(np.uint8)
            areas[f'{class_labels[cls]}'] += np.sum(mask)
            if mask[image_center[1], image_center[0]]:
                features[f'{class_labels[cls]}_center'] = 1
    for key, value in areas.items():
        features[f'{key}_area'] = value / total_area

    # Step 2: Check for intersections
    for i, cls in enumerate(classes):
        features[f'{class_labels[cls]}_total_count'] += 1

    # Exit if no intersections
    if features['intersection_total_count'] == 0:
        return features

    # Step 3: Extract intersection contours and find the main intersection
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
                if M['m00'] == 0:
                    continue
                else:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                distance_to_center = np.linalg.norm(np.array([cx, cy]) - np.array(image_center))
                # Assign to main intersection if it's the closest to the center
                if distance_to_center < min_distance_to_center:
                    min_distance_to_center = distance_to_center
                    main_intersection = {
                        'mask': mask, 'contour': contour, 'center': (cx, cy),
                        'box': cv2.boxPoints(cv2.minAreaRect(contour))
                    }
    
    # Step 4: Add main intersection dimensions to features dict
    quad_points = None
    if main_intersection:
        contour = main_intersection['contour']
        # Ensure contour is a 2D array of (x, y) points
        if contour.ndim == 3:
                contour = contour[:, 0, :]  # Convert from (N, 1, 2) to (N, 2)
        # Fit polygon, measure sides and assign directions
        polygon = fit_polygon_delaunay(contour)
        main_intersection['polygon'] = polygon
        side_lengths, quad_points = measure_intersection_sides(polygon)
        for i, length in enumerate(side_lengths, start=1):
            features[f'int_side{i}_length'] = length
        # Add area to features dict
        features['intersection_main_area'] = np.sum(main_intersection['mask'])

    # Step 5: Identify crosswalks at main intersection and measure dimensions
    crosswalk_counter = 0
    crosswalk_polygons = {}
    for i, cls in enumerate(classes):
        if class_labels[cls] == 'crosswalk':
            # Extract mask
            mask = masks[i].astype(np.uint8)
            # Find the largest contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            # Ensure contour is a 2D array of (x, y) points
            if contour.ndim == 3:
                contour = contour[:, 0, :]  # Convert from (N, 1, 2) to (N, 2)
            # Check if the crosswalk is close to the main intersection
            close_to_intersection = False
            if main_intersection and main_intersection['contour'] is not None:
                for point in contour:
                    point_tuple = (int(point[0]), int(point[1]))
                    distance = cv2.pointPolygonTest(main_intersection['contour'], point_tuple, True)
                    if abs(distance) <= cw_dist_thrshold:  # Check if within certain number of pixels
                        close_to_intersection = True
                        break  # No need to check further
            # If close to intersection, add to main_int_crosswalks
            if close_to_intersection:
                polygon = fit_polygon_delaunay(contour)
                crosswalk_polygons[crosswalk_counter] = polygon
                crosswalk_counter += 1
                features['crosswalk_main_count'] += 1
    # Assign crosswalks to sides and add to features
    if quad_points is not None:
        side_mapping = assign_crosswalks_to_sides(quad_points, crosswalk_polygons.values())
        for side_index, crosswalks in side_mapping.items():
            total_length = 0
            max_width = 0
            for crosswalk_polygon in crosswalks:
                length, width = measure_crosswalk_dimensions(crosswalk_polygon)
                total_length += length
                max_width = max(max_width, width)
            features[f'cw_side{side_index}_length'] = total_length
            features[f'cw_side{side_index}_width'] = max_width

    # Step 6: Count other road features
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
            if main_intersection and main_intersection['contour'] is not None:
                for point in contour:
                    point_tuple = (int(point[0]), int(point[1]))
                    distance = cv2.pointPolygonTest(main_intersection['contour'], point_tuple, True)
                    if abs(distance) <= other_dist_thrshold:  # Check if within certain number of pixels
                        close_to_intersection = True
                        features[f'{class_labels[cls]}_total_count'] += 1
                        break  # No need to check further
            # If close to intersection, add to main_int_crosswalks
            if close_to_intersection:
                features[f'{class_labels[cls]}_main_count'] += 1

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
    

def visualize_intersection_sides(predictor, img_file_path, metadata, class_labels, fit_function, plot=True):
    '''
    Visualize the four sides of an intersection polygon, assigning lengths and indexed sides.

    Parameters:
        predictor: Detectron2 predictor object.
        img_file_path (str): Path to the image file.
        metadata: Metadata object for visualizer.
        class_labels: List of class labels corresponding to class indices.
        fit_function: Function to fit polygons, e.g., `fit_polygon_delaunay`.
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

            # Measure sides and get quad_points
            try:
                side_lengths, quad_points = measure_intersection_sides(polygon)
            except ValueError as e:
                print(f'Warning: {e}')
                continue  # Skip this intersection if approximation fails

            # Draw the quadrilateral
            for j in range(len(quad_points)):
                pt1 = tuple(quad_points[j])
                pt2 = tuple(quad_points[(j + 1) % len(quad_points)])  # Wrap around
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)  # Green polygon edges

            # Annotate side indices and lengths
            for index, length in enumerate(side_lengths, start=1):
                pt1 = quad_points[index - 1]
                pt2 = quad_points[index % len(quad_points)]  # Wrap around
                midpoint = tuple(((np.array(pt1) + np.array(pt2)) / 2).astype(int))
                cv2.putText(img, f'Side {index}: {length:.2f}',
                            midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if plot:
        # Show the results
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

def visualize_crosswalk_measurements(predictor, img_file_path, metadata, class_labels, fit_function, plot=True):
    '''
    Visualize crosswalk measurements (length and width) based on the fitted polygon.

    Parameters:
        predictor: Detectron2 predictor object.
        img_file_path (str): Path to the image file.
        metadata: Metadata object for the visualizer.
        class_labels: List of class labels corresponding to class indices.
        fit_function: Polygon fitting function (e.g., fit_polygon_delaunay).
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
            cv2.putText(img, f'Len: {length:.2f}', (midpoint[0] - 50, midpoint[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(img, f'Wid: {width:.2f}', (midpoint[0] - 50, midpoint[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if plot:
        # Show the results
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
    img_crosswalk_measurements = visualize_crosswalk_measurements(predictor, img_file_path, metadata, class_labels, fit_function, plot=False)

    # Create a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    # Add images to the grid
    axs[0, 0].imshow(img_annotations) # img_predictions
    axs[0, 0].set_title('Manual Annotations')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(img_predictions) # img_intersection
    axs[0, 1].set_title('Model Predictions')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(img_intersection) # img_crosswalk_directions
    axs[1, 0].set_title('Intersection Dimensions')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(img_crosswalk_measurements)
    axs[1, 1].set_title('Crosswalk Measurements')
    axs[1, 1].axis('off')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    

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
    if dataset_dicts:
        img_annotations = visualize_annotations(image_file_path, metadata, dataset_dicts, plot=False)
    else:
        img = cv2.imread(image_file_path)
        img_annotations = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_annotations = visualize_annotations(image_file_path, metadata, dataset_dicts, plot=False)
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


# Storage of predictions
def predictions_to_dict(predictions):
    pred_boxes = predictions['instances'].pred_boxes.tensor.cpu().numpy().tolist()
    pred_scores = predictions['instances'].scores.cpu().numpy().tolist()
    pred_classes = predictions['instances'].pred_classes.cpu().numpy().tolist()
    
    # Convert masks to RLE (Run-Length Encoding) for compact storage
    pred_masks = [mask.cpu().numpy() for mask in predictions['instances'].pred_masks]
    rle_masks = [mask_util.encode(np.array(mask[:, :, None], order="F"))[0] for mask in pred_masks]
    
    # Convert RLE counts from bytes to strings
    for rle in rle_masks:
        rle['counts'] = rle['counts'].decode('utf-8')
    
    return {
        "boxes": pred_boxes,
        "scores": pred_scores,
        "classes": pred_classes,
        "masks": rle_masks,
    }

def reconstruct_predictions_from_dict(prediction_dict, image_dims):
    """
    Reconstructs a Detectron2 Instances object from prediction data.
    """
    # Convert RLE masks back to binary masks
    from pycocotools import mask as mask_util
    rle_masks = prediction_dict['masks']
    pred_masks = [mask_util.decode(rle) for rle in rle_masks]
    
    # Create tensor for masks
    pred_masks_tensor = torch.tensor(pred_masks, dtype=torch.bool)

    # Create tensors for boxes, classes, and scores
    pred_boxes = torch.tensor(prediction_dict['boxes'], dtype=torch.float32)
    pred_classes = torch.tensor(prediction_dict['classes'], dtype=torch.int64)
    pred_scores = torch.tensor(prediction_dict['scores'], dtype=torch.float32)

    # Construct Instances object
    instances = Instances(
        image_size=image_dims,
        pred_boxes=Boxes(pred_boxes),
        pred_classes=pred_classes,
        scores=pred_scores,
        pred_masks=pred_masks_tensor,
    )
    return {"instances": instances}  # Return a dictionary with the instances

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
    image_files = os.listdir(IMAGES_DIR) # [x for x in os.listdir(IMAGES_DIR) if x.replace('.png', '.json') in os.listdir(ANNOTATIONS_DIR)]
    if NUM_IMAGES_TO_PROCESS:
        image_files = image_files[:NUM_IMAGES_TO_PROCESS]

    print(f'Processing {len(image_files)} images...')
    
    # Extract features
    all_features = []
    start = time.time()
    for i, image_file in enumerate(image_files):
        # Open the image file
        image_file_path = os.path.join(IMAGES_DIR, image_file)
        image = cv2.imread(image_file_path)
        # Generate predictions and extract features
        predictions = predictor(image)
        features = extract_features(image_file_path, predictions, class_labels)
        features['image_name'] = image_file.replace('.png', '')
        # Add features to the list
        all_features.append(features)
        # Save each prediction to a JSON file
        prediction_dict = predictions_to_dict(predictions)
        image_name = image_file.replace('.png', '')
        output_file = os.path.join(PREDICTIONS_DIR, f'{image_name}.json')
        with open(output_file, 'w') as f:
            json.dump(prediction_dict, f)
        # Print progress
        if i % 1000 == 0 and i > 0:
            print(f'Processed {i} images...Elapsed time: {(time.time()-start)/60:.2f} minutes')

    print(f'Finished processing {len(image_files)} images. Elapsed time: {(time.time()-start)/60:.2f} minutes')
    
    # Save results to a DataFrame
    df = pd.DataFrame(all_features)
    output_file = os.path.join(OUTPUT_DIR, f'image_features_{datetime.now().strftime("%m.%d_%H.%M")}.csv')
    df.to_csv(output_file, index=False)