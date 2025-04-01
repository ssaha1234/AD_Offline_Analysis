#!/usr/bin/env python
# coding: utf-8

# In[1]:


#conda activate l1AD_env

import numpy as np
import matplotlib.pyplot as plt
import h5py
#import random
#import sklearn
#import collections
#from sklearn.model_selection import train_test_split
#import json
#import pylab 
#from scipy.optimize import curve_fit
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
import matplotlib
import matplotlib.patches as mpatches
#import shap
#import pandas as pd
import tensorflow as tf
#import tarfile
from tensorflow.keras.models import load_model
#from qkeras import QActivation, QDense, QConv2D, QBatchNormalization
import ensembler_functions_offline as ef
#import tf2onnx
#import onnx
import os
from sklearn.model_selection import train_test_split
#import load_and_match as lam

import sklearn
print(sklearn.__version__)
print('tensorflow',tf.__version__)

import sys
print('python',sys.version)

print("matplotlib version:", matplotlib.__version__)
print("h5py version:", h5py.__version__)
print("numpy version:", np.__version__)




# In[20]:


#import sys
#sys.executable
#!{sys.executable} -m pip install qkeras


# In[2]:


# Set matplotlib default color cycle
new_color_cycle = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
    '#aec7e8',
    '#ffbb78',
    '#98df8a',
    '#ff9896',
    '#c5b0d5',
    '#c49c94',
    '#f7b6d2',
    '#c7c7c7',
    '#dbdb8d',
    '#9edae5'
]

# You can then apply this new color cycle to your matplotlib plots
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=new_color_cycle)


# In[3]:


L1AD_rate = 1000
target_rate = 10


# In[23]:


# Load and match my data with the topo2A (L1AD) scores. Only needs to be done once ever.
#lam.load_and_match('./h5_ntuples/11-5-2024')
#%run load_and_match.py

#lam.load_and_match('/eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/ntuples/EB_h5_10-06-2024')


# In[4]:


#import os

#file_path = '/eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/ntuples/trained_models/multiple_trainings/trial_8/EB_HLT_0.keras'
#os.path.exists(file_path)

#get_ipython().system('file /eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/ntuples/trained_models/multiple_trainings/trial_8/EB_HLT_0.keras')


#from tensorflow.keras.models import load_model

#model = load_model('/eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/ntuples/trained_models/multiple_trainings/trial_8/EB_HLT_0.keras')


# In[5]:

	
data_info = {
    "train_data_scheme": "topo2A_train", 
    "pt_normalization_type": "global_division", 
    "L1AD_rate": 1000,
    "pt_thresholds": [50, 20, 20, 20],
    "comments": "new L1AD model"
}


training_info = {
    "save_path": "/eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/ntuples/trained_models/multiple_trainings/trial_test_DisCo", 
    "dropout_p": 0.1, 
    "L2_reg_coupling": 0.01, 
    "latent_dim": 4, 
    "large_network": True,
    #Test with 1 training, originally set to 10
    "num_trainings": 1,
    "training_weights": True
}

datasets, data_info = ef.load_and_preprocess(**data_info)
print("LOADING AND PREPROCESS DONE")

# Can we skip this if we already have the trained files? - 
#If you do not have the keras files, you need to run this line, or you will get this erros - FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/ntuples/trained_models/multiple_trainings/trial_12/EB_HLT_0.keras', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

training_info, data_info = ef.train_multiple_models(datasets, data_info, **training_info)

# In[17]:


ef.process_multiple_models(
    training_info=training_info,
    data_info=data_info,
    plots_path=training_info['save_path']+'/plots_Feb21_epoch2',
    target_rate=target_rate,
    L1AD_rate=L1AD_rate
)








