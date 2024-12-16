# Assessing Intersection Safety in New York City Using Computer Vision
This repo provides the implementation of data collection and cleaning, model training and tuning and final inference for the thesis paper "Assessing Intersection Safety in New York City Using Computer Vision," by Keith Colella. The paper was written for completion of CUNY'S Masters in Data Science Program. The abstract is shown below.

*<div align="center">This research examines how physical infrastructure affects pedestrian and cyclist safety at more than 27,000 intersections in New York City. A fine-tuned Mask R-CNN model extracts infrastructure features from Google Maps imagery. These features are used in ordinal regression models to analyze NYPD collision data. Results show a significant positive effect of intersection complexity on collision frequency for both cyclists (\beta = 2.882, p < 0.001) and pedestrians (\beta = 4.117, p < 0.001). Specialized infrastructure like bike lanes and crosswalks correlate positively with their respective accident types, but this correlation likely reflects the influence of these features on route choice rather than safety. The impact of complexity on collision frequency decreases with appropriate lane markings, as indicated by significant negative interaction terms in both cyclist (\beta = -3.869, p < 0.001) and pedestrian (\beta = -4.508, p < 0.001) models. These findings suggest that strategic placement of specialized infrastructure away from large, complex intersections could improve traffic safety outcomes.</div>*  

### Scripts  
The scripts are organized as follows.  

• `rev_geocode_intersections.py` - Takes a unique list of latitude / longitude pairs from NYPD collision data and performs reverse geocoding using the GeoNames API. It used the `findNearestIntersection` call to associate latitude / longitude pairs with distinct intersections.  
• `get_google_images.py` - Takes de-duped list of ~27,000 intersections and queries images from the Google Static Maps API. Images that were not previously queried or returned error messages (i.e., "Image not available") can be re-run.  
• `train_is_model.py` - Prepares data and trains the instance segmentation model. Hyperparameters can be controlled via `config.json`.  
• `hp_tuning.py` - Performs hyperparameter tuning to optimize training. Focus is on learning rate, batch size and max iterations.  
• `is_model_inference.py` - Implements the feature extraction pipeline to take raw model predictions and generate specific measurements of intersection characteristics. Functions are modular and implemented in other scripts (namely `5_error_analysis.ipynb`).  

For these scripts, key inputs are controlled via `config.json`

### Notebooks  
The notebooks primarily provide plots, visualizations, and iterative tests to explore data, analyze errors and tune models. The only two required for replication are the following:

• `1_intersections_deduping.ipynb` - Using the raw collision data from NYPD and coordinate pairs obtained from `rev_geocode_intersections.py`, this notebook takes the collision-level data and aggregates it at the intersection level. Over 1.6 million collisions are collected into ~27,000 intersections, and unique identifier based on latitude / longitude is created.  
• `7_model_diagnostic.ipynb` - Performs the final fitting of ordinal regression models using discretized, adjusted collision rates. Implements diagnostics, sensitivity analysis and stability testing to assess model robustness.  

Beyond that, the other notebooks serves secondary purposes, but are not strictly required for replication.

• `0_collisions_cleaning.ipynb` - Initial EDA and cleaning of NYPD collision data.  
• `2_merging_int_coll.ipynb` - Merges the de-duped intersection list with additional features from the NYPD collision data.  
• `3_is_feature_extraction.ipynb` - Testing notebook for feature extraction functions that are ultimately refined for the pipeline in `is_model_inference.py`.  
• `4_sampling.ipynb` - Performs targeted samples of images for annotation to support training of the instance segmentation model.  
• `5_error_analysis.ipynb` - Constructs visualizations of model annotations versus predictions to assess patterns in model errors.  
• `6_feature_eda.ipynb` - Initial creation of explanatory variables and fitting of regression models for final conclusions. Leads to choices are final ordinal regression models, implemented in `7_model_diagnostic.ipynb`.  