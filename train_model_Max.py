# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
import matplotlib
import matplotlib.patches as mpatches
import tensorflow as tf
#import tf2onnx
#import onnx
import os
from sklearn.model_selection import train_test_split
import load_and_match as lam
import ensembler_functions as ef

# Set matplotlib default color cycle
new_color_cycle = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
    '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', 
    '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]


# Apply the new color cycle to matplotlib plots
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=new_color_cycle)

# Define constants
L1AD_rate = 1000
target_rate = 10
print('setup done')
print('running load and match')
# Load and match data (this is equivalent to running the Jupyter cell with %run)

#lam.load_and_match('/eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/EB_datasets_28102024/EB_h5_10-06-2024')

# Load ensembler functions (assuming it's a Python file, not a Jupyter notebook)
# Replace the `%run ensembler_functions.py` with importing the functions directly if it's a separate Python script
# If `ensembler_functions.py` is a script, ensure that it contains functions to be imported
# For this conversion, we assume it's already imported above as `import ensembler_functions as ef`

# Print keys (for debugging, if required)
print("Data Info Keys:", list(ef.load_and_preprocess.__code__.co_varnames))  # Example to see function args

# Configuration for data and training
data_info = {
    "train_data_scheme": "topo2A_train+overlap", 
    "pt_normalization_type": "StandardScaler", 
    "L1AD_rate": 1000
}

training_info = {
    "save_path": "/eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/ntuples/trained_models/multiple_trainings/trial_8", 
    "dropout_p": 0.1, 
    "L2_reg_coupling": 0.01, 
    "latent_dim": 4, 
    "large_network": True,
    "num_trainings": 1,  # Test with 1 training, originally set to 10
    "training_weights": True
}

# Load and preprocess datasets
datasets, data_info = ef.load_and_preprocess(**data_info)

# Train multiple models
training_info, data_info = ef.train_multiple_models(datasets, data_info, **training_info)

# Process multiple models
ef.process_multiple_models(
    training_info=training_info,
    data_info=data_info,
    plots_path=os.path.join(training_info['save_path'], 'plots'),
    target_rate=target_rate,
    L1AD_rate=L1AD_rate
)