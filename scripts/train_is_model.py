# Core
import random
import json
import os
import copy

# COCO
import labelme2coco

# Detectron2 Data
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, build_detection_train_loader
from detectron2.data.transforms import RandomFlip, ResizeShortestEdge, RandomBrightness, RandomContrast
from detectron2.data.transforms import RandomSaturation, RandomLighting, RandomCrop, RandomRotation

# Detectron2 Training
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo

# Detectron2 Evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Warning and logs
import warnings
import logging


# Ensure reproducibility
random.seed(42)

# Training Config
with open('config.json', 'r') as f:
    config = json.load(f)['training']

IMAGES_DIR = config['images_dir']
ANNOTATIONS_DIR = config['annotations_dir']
MODELS_DIR = config['models_dir']
OUTPUT_SPLIT_FILE = config['output_split_file']
EXCLUDED_CLASSES = config['excluded_classes']
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = config['ratios']

# Model hyperparameters
with open('config.json', 'r') as f:
    config = json.load(f)['hyperparameters']

NUM_WORKERS = config['num_workers']
BASE_LR = config['base_lr']
MAX_ITER = config['max_iter']
IMS_PER_BATCH = config['ims_per_batch']
BATCH_SIZE_PER_IMAGE = config['batch_size_per_image']
DATA_AUGMENTATION = config['data_augmentation']


def prep_annotations():
    '''
    Convert Labelme annotations to aggregated COCO format.
    '''
    # Ensure image names are all up to date
    for json_file in os.listdir(ANNOTATIONS_DIR):
        if json_file.endswith('.json'):
            json_path = os.path.join(ANNOTATIONS_DIR, json_file)
            # Load the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            # Update the imagePath field
            image_name = os.path.splitext(json_file)[0] + '.png'  # Assuming images are PNGs
            data['imagePath'] = os.path.join(IMAGES_DIR, image_name)
            # Save the updated JSON
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
    
    # Ensure previous aggregated files are removed
    aggregated_files = [f for f in os.listdir(ANNOTATIONS_DIR) if '_' not in f]
    for f in aggregated_files:
        os.remove(os.path.join(ANNOTATIONS_DIR, f))

    # Convert Labelme annotations to COCO
    labelme2coco.convert(ANNOTATIONS_DIR, ANNOTATIONS_DIR)

    # Open the COCO annotations file
    annotation_file = os.path.join(ANNOTATIONS_DIR, 'dataset.json')
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Fix file_name values in aggregated json
    for image in coco_data['images']:
        image['file_name'] = os.path.basename(image['file_name'])

    # Remove annotations for excluded classes
    excluded_ids = [x['id'] for x in coco_data['categories'] if x['name'] in EXCLUDED_CLASSES]
    coco_data['annotations'] = [ann for ann in coco_data['annotations'] if ann['category_id'] not in excluded_ids]
    coco_data['categories'] = [cat for cat in coco_data['categories'] if cat['id'] not in excluded_ids]
    
    # Save the fixed annotations
    with open(annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f'Aggregated and cleaned annotations saved to {annotation_file}')


def create_splits():
    '''
    Split dataset into train, val, and test sets and save mapping.
    '''
    # Get all annotated files
    annotation_files = [
        f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.json') and '_' in f
    ]
    # Get only images that have associated annotations
    image_files = [
        f.replace('.json', '.png') for f in annotation_files 
        if os.path.exists(os.path.join(IMAGES_DIR, f.replace('.json', '.png')))
    ]

    # Shuffle and split
    random.shuffle(image_files)
    total_images = len(image_files)
    train_end = int(TRAIN_RATIO * total_images)
    val_end = train_end + int(VAL_RATIO * total_images)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    splits = {'train': train_files, 'val': val_files, 'test': test_files}

    # Save mapping
    with open(OUTPUT_SPLIT_FILE, 'w') as f:
        json.dump(splits, f, indent=4)

    # Load COCO data and splits
    with open('data/annotations/dataset.json', 'r') as f:
        coco_data = json.load(f)

    # Create annotation files for each split
    for split, images in splits.items():
        filtered_coco = copy.deepcopy(coco_data)
        # Filter images
        images = [img for img in coco_data['images'] if os.path.basename(img['file_name']) in images]
        image_ids = [img['id'] for img in images]
        # Filter annotations
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
        # Update coco and register
        filtered_coco['images'] = images
        filtered_coco['annotations'] = annotations
        # Save as json
        with open(f'data/annotations/{split}.json', 'w') as f:
            json.dump(filtered_coco, f, indent=4)

    print(
        f'Split mapping saved to {OUTPUT_SPLIT_FILE}, with',
        f'train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}'
    )


def train_model(
        base_lr=0.0025, max_iter=100, ims_per_batch=8, batch_size_per_image=64, num_workers=12, 
        data_augmentation=False, models_dir=False, print_output=False
    ):
    '''
    Train model.
    '''
    # Set up logger and handle outputs
    if not print_output:
        logger = setup_logger(output='logs') # None
        logger.setLevel(logging.CRITICAL)
        # Disable logging from other libraries
        logging.getLogger('detectron2').setLevel(logging.CRITICAL)
        logging.getLogger('fvcore').setLevel(logging.CRITICAL)
        logging.getLogger('iopath').setLevel(logging.CRITICAL)
        # Ignore warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='torch.functional')
        warnings.filterwarnings('ignore', module='fvcore.common.checkpoint')

    # Load COCO data and splits
    with open('data/annotations/dataset.json', 'r') as f:
        coco_data = json.load(f)
    class_names = [x['name'] for x in coco_data['categories']]

    with open('data/split_mapping.json', 'r') as f:
        splits = json.load(f)

    # Register datasets
    for split, images in splits.items():
        annotations_file = os.path.join(ANNOTATIONS_DIR, f'{split}.json')
        register_coco_instances(split, {}, annotations_file, IMAGES_DIR)
        DatasetCatalog.get(split)
        MetadataCatalog.get(split)

    # Load base config and set model architecture
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'))

    # Set dataset
    cfg.DATASETS.TRAIN = ('train',)
    cfg.DATASETS.VAL = ('val',) if len(splits['val']) else ()
    cfg.DATASETS.TEST = ('test',) if len(splits['test']) else ()
    cfg.DATALOADER.NUM_WORKERS = num_workers

    # Use pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml')

    # Training parameters for quick trial
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    print(f"IMS_PER_BATCH: {ims_per_batch}, Type: {type(ims_per_batch)}")

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)  # Number of custom classes
    cfg.MODEL.DEVICE = 'cuda'

    # Output directory
    cfg.OUTPUT_DIR = models_dir if models_dir else './models' # './models'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Data augmentation
    if data_augmentation:
        augmentations = [
            ResizeShortestEdge(short_edge_length=(640, 800), max_size=1333, sample_style="choice"),
            RandomFlip(prob=0.5, horizontal=True, vertical=False),
            RandomBrightness(0.8, 1.2),  # Adjust brightness by a factor in the range [0.8, 1.2]
            RandomContrast(0.8, 1.2),    # Adjust contrast by a factor in the range [0.8, 1.2]
            RandomSaturation(0.8, 1.2),  # Adjust saturation by a factor in the range [0.8, 1.2]
            RandomLighting(0.1),         # Add lighting noise
            RandomCrop(crop_type="absolute_range", crop_size=(512, 512)), # Randomly crop to a 512x512 region
            RandomRotation(angle=[-15, 15]), # Rotate randomly between -15 to 15 degrees
        ]

    else:
        augmentations = []

    # Train loader
    mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
    train_loader = build_detection_train_loader(cfg, mapper=mapper)

    # Train the model
    trainer = DefaultTrainer(cfg)
    trainer.data_loader = train_loader
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation after training - only if val available
    if len(splits['val']): 
        evaluator = COCOEvaluator(
            dataset_name='val', tasks=('bbox', 'segm'),  
            distributed=False, output_dir=os.path.join(cfg.OUTPUT_DIR, 'eval')
        )
        val_loader = build_detection_test_loader(cfg, 'val')
        results = inference_on_dataset(trainer.model, val_loader, evaluator)
        return results

    else:
        return None

def evaluate_model():
    '''
    Evaluate model.
    '''
    pass


### MAIN
if __name__ == '__main__':
    prep_annotations()
    create_splits()
    train_model(
        base_lr=BASE_LR, 
        max_iter=MAX_ITER, 
        batch_size_per_image=BATCH_SIZE_PER_IMAGE, 
        ims_per_batch=IMS_PER_BATCH,
        num_workers=NUM_WORKERS,
        data_augmentation=DATA_AUGMENTATION,
        models_dir=MODELS_DIR,
        print_output=True
    )
    evaluate_model()
