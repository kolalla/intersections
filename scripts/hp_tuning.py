import pandas as pd
import numpy as np
import os
import json
import optuna
import cv2
from datetime import datetime
from matplotlib import pyplot as plt
from train_is_model import prep_annotations, create_splits, train_model
from is_model_inference import setup_predictor, extract_features, annotations_to_model_format
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# Config parameters
LOG_FILE = f'logs/hp_tuning_results_{datetime.now().strftime("%m.%d_%H.%M")}.csv'
N_TRIALS = 10
MODELS_DIR = 'models'
MODEL_NAME = 'model_final.pth'
IMAGES_DIR = 'data/images'
ANNOTATIONS_DIR = 'data/annotations'

# Search space
with open('config.json', 'r') as f:
    config = json.load(f)['hp_space']
BASE_LR_MIN = config['base_lr']['min']
BASE_LR_MAX = config['base_lr']['max']
BASE_LR_LOG = config['base_lr']['log']
MAX_ITER_MIN = config['max_iter']['min']
MAX_ITER_MAX = config['max_iter']['max']
BATCH_SIZE_PER_IMAGE_MIN = config['batch_size_per_image']['min']
BATCH_SIZE_PER_IMAGE_MAX = config['batch_size_per_image']['max']
IMS_PER_BATCH_MIN = config['ims_per_batch']['min']
IMS_PER_BATCH_MAX = config['ims_per_batch']['max']

# Functions
def compare_feature_dicts(ground_truth, predictions):
    '''
    Compare two feature dictionaries (ground truth and predictions) using absolute percentage error (APE) for all features.
    '''
    results_list = []

    for key in ground_truth:
        if key in predictions:
            gt_value = ground_truth[key]
            pred_value = predictions[key]

            # Handle None or empty values
            if not gt_value and pred_value:
                gt_value = 0
            if not pred_value and gt_value:
                pred_value = 0
            
            # Skip zero-zero rows
            if gt_value == 0 and pred_value == 0:
                continue
            
            # Compute absolute percentage error
            if gt_value == 0:
                ape = min(1, abs(pred_value))  # Cap APE at 100% for zero ground truth
            else:
                ape = abs(gt_value - pred_value) / (abs(gt_value) + 1e-6)

            # Initialize row dictionary
            row = {
                'feature': key,
                'ground_truth': gt_value,
                'predicted': pred_value,
                'ape': ape
            }

            # Add row to results
            results_list.append(row)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    # Compute aggregated accuracy
    aggregated_accuracy = max(0, min(1, 1 - results_df['ape'].mean()))  # Clamp between 0 and 1

    return results_df, aggregated_accuracy


def get_aggregated_accuracy(annotated_images, dataset_dicts, predictor, class_names, images_dir, plot=True):
    aggr_accuracies = {}
    for image_name in annotated_images:
        image_file_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_file_path) 
        dataset_dict = [x for x in dataset_dicts if x['file_name'] == image_file_path]
        if dataset_dict:
            dataset_dict = dataset_dict[0]
        else:
            continue
        annotations_pred = annotations_to_model_format(dataset_dict)
        model_pred = predictor(image)

        annotation_features = extract_features(image_file_path, annotations_pred, class_names)
        predicted_features = extract_features(image_file_path, model_pred, class_names)
        _, aggregated_accuracy = compare_feature_dicts(annotation_features, predicted_features)

        aggr_accuracies[image_name] = aggregated_accuracy

    if plot:
        plt.figure(figsize=(7, 5))
        plt.hist([x for x in aggr_accuracies.values()])
        plt.xlabel('Aggregated accuracy')
        plt.show()

    return aggr_accuracies


def log_trial_results(trial_id, hyperparameters, metrics, per_category_ap, log_file=LOG_FILE):
    """
    Logs trial results to a CSV file.

    :param trial_id: Identifier of the trial
    :param hyperparameters: Dict of hyperparameter values (e.g., {'base_lr': 0.01, 'max_iter': 200})
    :param metrics: Dict of overall metrics (e.g., {'mAP': 8.7, 'AP50': 12.3})
    :param per_category_ap: Dict of per-category AP values (e.g., {'intersection': 0.1, 'crosswalk': 0.3})
    :param log_file: Path to the CSV file for storing logs
    """
    row = {
        'trial_id': trial_id,
        **hyperparameters,
        **metrics,
    }
    for category, ap in per_category_ap.items():
        row[f'ap_{category}'] = ap

    # Append the row to CSV
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
    else:
        df = pd.DataFrame()
    
    # Add new row
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(log_file, index=False)


def objective(trial):
    # Define search space
    base_lr = trial.suggest_float('base_lr', BASE_LR_MIN, BASE_LR_MAX, log=BASE_LR_LOG)
    max_iter = trial.suggest_int('max_iter', MAX_ITER_MIN, MAX_ITER_MAX)
    batch_size_per_image = trial.suggest_int('batch_size_per_image', BATCH_SIZE_PER_IMAGE_MIN, BATCH_SIZE_PER_IMAGE_MAX)
    ims_per_batch = trial.suggest_int('ims_per_batch', IMS_PER_BATCH_MIN, IMS_PER_BATCH_MAX)
    augmentation = trial.suggest_categorical('augmentation', [True, False])

    # Print trial parameters
    print(
        f'----Parameters for trial #{trial.number}----\n'
        f'  base_lr: {base_lr}\n'
        f'  max_iter: {max_iter}\n'
        f'  batch_size_per_image: {batch_size_per_image}\n'
        f'  ims_per_batch: {ims_per_batch}\n'
        f'  augmentation: {augmentation}\n'
    )

    # Call train_model with trial parameters
    results = train_model(
        base_lr=base_lr, 
        max_iter=max_iter, 
        batch_size_per_image=batch_size_per_image, 
        ims_per_batch=ims_per_batch,
        data_augmentation=augmentation,
        models_dir=MODELS_DIR,
        print_output=False
    )

    # Extract AP metrics from training results
    mAP_segm = results['segm']['AP']  # Overall segmentation mAP
    AP50 = results['segm']['AP50']    # AP at IoU=0.50
    per_category_ap = {k:v for k, v in results['segm'].items() if 'AP-' in k}  # Per-category APs (dict)

    # Register validation dataset
    annotations_file = os.path.join(ANNOTATIONS_DIR, 'val.json')
    register_coco_instances('val', {}, annotations_file, IMAGES_DIR)
    dataset_dicts = DatasetCatalog.get('val')
    metadata = MetadataCatalog.get('val')
    annotated_images = [os.path.basename(x['file_name']) for x in dataset_dicts]

    # Setup model
    model_file_path = os.path.join(MODELS_DIR, MODEL_NAME)
    class_names = metadata.thing_classes
    num_classes = len(class_names)
    predictor = setup_predictor(model_file_path, num_classes, threshold=0.7)

    # Get aggregated accuracy
    aggregated_accuracies = get_aggregated_accuracy(
        annotated_images, dataset_dicts, predictor, class_names, IMAGES_DIR, plot=False
    )
    avg_accuracy = np.mean(list(aggregated_accuracies.values()))

    # Log trial results
    hyperparameters = {
        'base_lr': base_lr,
        'max_iter': max_iter,
        'batch_size_per_image': batch_size_per_image,
        'ims_per_batch': ims_per_batch,
        'augmentation': augmentation,
    }
    metrics = {
        'avg_accuracy': avg_accuracy,
        'mAP': mAP_segm,
        'AP50': AP50,
    }
    log_trial_results(trial.number, hyperparameters, metrics, per_category_ap)

    # Use average accuracy as the optimization metric
    return avg_accuracy

# Main
if __name__ == "__main__":
    # Ensure data is prepped
    prep_annotations()
    create_splits()

    # Create Optuna study
    study = optuna.create_study(direction='maximize')  # Maximize mAP
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    # Print best parameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")