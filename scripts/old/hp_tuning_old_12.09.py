import pandas as pd
import os
import json
import optuna
from datetime import datetime
from train_is_model import prep_annotations, create_splits, train_model

# Config parameters
LOG_FILE = f'logs/hp_tuning_results_{datetime.now().strftime("%m.%d_%H.%M")}.csv'
N_TRIALS = 50
MODELS_DIR = 'models'

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

    # Extract metrics
    mAP_segm = results['segm']['AP']  # Overall segmentation mAP
    AP50 = results['segm']['AP50']    # AP at IoU=0.50
    per_category_ap = {k:v for k, v in results['segm'].items() if 'AP-' in k}  # Per-category APs (dict)

    # Log trial results
    hyperparameters = {
        'base_lr': base_lr,
        'max_iter': max_iter,
        'batch_size_per_image': batch_size_per_image,
        'ims_per_batch': ims_per_batch,
        'augmentation': augmentation,
    }
    metrics = {
        'mAP': mAP_segm,
        'AP50': AP50,
    }
    log_trial_results(trial.number, hyperparameters, metrics, per_category_ap)
    
    # Use segmentation mAP as the optimization metric
    mAP_segm = results['segm']['AP']  # Adjust based on Detectron2 output format
    return mAP_segm

# Main
if __name__ == "__main__":
    # Ensure data is prepped
    prep_annotations()
    create_splits()

    # Create Optuna study
    study = optuna.create_study(direction='maximize')  # Maximize mAP
    study.optimize(objective, n_trials=N_TRIALS)

    # Print best parameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")