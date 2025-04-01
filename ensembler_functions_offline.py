import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras import layers, Model
import json
import matplotlib.pyplot as plt
#import tf2onnx
#import onnx
#import onnxruntime as rt
#import ROOT
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------------------
def load_subdicts_from_h5(save_dir):
    """
    Loads sub-dictionaries of NumPy arrays from HDF5 files in a directory and reconstructs the original structure.
    
    Args:
        save_dir (str): The directory where the HDF5 files are stored.
    
    Returns:
        main_dict (dict): A dictionary of dictionaries where the innermost values are NumPy arrays.
    """
    main_dict = {}
    
    for filename in os.listdir(save_dir):
        if filename.endswith(".h5"):
            sub_dict_name = os.path.splitext(filename)[0]
            file_path = os.path.join(save_dir, filename)
            with h5py.File(file_path, 'r') as f:
                sub_dict = {key: np.array(f[key]) for key in f}
            main_dict[sub_dict_name] = sub_dict
            print(f"Loaded {sub_dict_name} from {file_path}")
    
    return main_dict
# -----------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------
def combine_data(datasets, tags_to_combine, new_tag):
    """
    Combines subdicts of the 'datasets' dict.
    
    Inputs: 
        datasets: dict that maps {dataset_tag : dataset_dict}
        tags_to_combine: list if strings [dataset_tag1, ..., dataset_tagN] of the tags to be combined
        new_tag: the name of the new tag of the combined subdict

    Returns: 
        datasets: same datasets dict as input, but with the specified tags combined.
    """

    # initialize empty lists for new tag
    datasets[new_tag] = {key: [] for key in datasets[tags_to_combine[0]].keys()}

    # Loop through old tags and append np arrays to lists
    for tag in tags_to_combine:
        for key, value in datasets[tag].items():
            datasets[new_tag][key].append(value)

    # Concatenate lists into single np array
    for key, value in datasets[new_tag].items():
        datasets[new_tag][key] = np.concatenate(value, axis=0)

    # Delete old tags
    for tag in tags_to_combine:
        del datasets[tag]

    # Make sure everything is an np array
    for tag, data_dict in datasets.items():
        for key, value in data_dict.items():
            data_dict[key] = np.array(value)

    return datasets
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
class ZeroAwareStandardScaler:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        # Initialize means and stds arrays with same shape as features
        n_features = X.shape[1]
        self.means = np.zeros(n_features)
        self.stds = np.zeros(n_features)

        # Compute mean and std for each feature, ignoring zero values
        for i in range(n_features):
            non_zero_values = X[:, i][X[:, i] != 0]
            if len(non_zero_values) > 0:
                self.means[i] = np.mean(non_zero_values)
                self.stds[i] = np.std(non_zero_values)
            else:
                self.means[i] = 0
                self.stds[i] = 1  # Avoid division by zero for empty or all-zero columns

    def transform(self, X):
        # Copy X to avoid changing the original dataset
        X_scaled = np.copy(X)

        # Apply scaling only to non-zero values
        for i in range(X.shape[1]):
            # Select non-zero values for feature i
            non_zero_mask = X[:, i] != 0
            X_scaled[non_zero_mask, i] = (X[non_zero_mask, i] - self.means[i]) / self.stds[i]

        return X_scaled
# -----------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------
def dphi(phi1, phi2):
    return np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi # accounts for periodicity of phi
# -----------------------------------------------------------------------------------------


def find_threshold(scores, weights, hlt_pass, target_rate, incoming_rate):
    """
    calculates the threshold and total rate for a specified incoming rate and target pure rate.

    Inputs:
        scores: array of AD scores
        weights: array of sample weights corresponding to the AD scores
        hlt_pass: list of True/False if each event passed the existing HLT. Used for pure rate calculation.
        target_rate: target pure rate that we want to run at
        incoming rate: incoming rate seen by the specified trigger.

    Returns:
        threshold, total_rate
        threshold: the number for which AD_score>=threshold is anomalous.
        total_rate: the total rate corresponding to the specified pure rate.
    """
    sorted_indices = np.argsort(scores)[::-1]
    cumulative_rate = np.cumsum(weights[sorted_indices] * (~hlt_pass[sorted_indices])) * incoming_rate / np.sum(weights)
    threshold_index = np.searchsorted(cumulative_rate, target_rate)
    total_rate = np.cumsum(weights[sorted_indices]) * incoming_rate / np.sum(weights)
    return scores[sorted_indices[threshold_index]], total_rate[threshold_index]

def calculate_efficiencies(anomalous, weights, hlt_pass):
    """
    Calculates signal efficiencies of the current HLT, as well as HLT with our AD algo added.

    Inputs:
        anomalous: array of 0s and 1s. 1 means the sample was classified as anomalous by the AD algo.
        weights: array of weights for each sample.
        hlt_pass: list of True/False if each event passed the existing HLT. Used for HLT rate calculation.

    Returns:
        hlt_eff: efficiency of current HLT (without our AD algo)
        combined eff: efficiency of current HLT + our AD algo.
    """
    hlt_eff = np.sum(hlt_pass * weights) / np.sum(weights)
    combined_eff = np.sum((hlt_pass | anomalous) * weights) / np.sum(weights)
    #ad_rate = incoming_rate * np.sum(weights[l1_seeded] * (anomalous[l1_seeded] & ~hlt_pass[l1_seeded])) / np.sum(weights[l1_seeded])
        
    return hlt_eff, combined_eff

def plot_efficiencies(results, bkg_type, save_path, target_rate=10, eff_type='ASE', seed_type='l1Seeded'):
    """
    Plotting code to see how much our algo adds on top of the current HLT for a variety of signals.

    Inputs:
        results: dict containing efficiency values. This will be created in future function, so this should not be worried about.
        bkg_type: either 'HLT' or 'L1' depending on the type of objects used.
        save_path: path in which to save the plots.
        target_rate: target pure rate for the HLTAD algo.
        eff_type: either 'ASE' or 'TSE' depending on the type of efficiency to be plotted. See int note for more details.
        seed_type: string containing the seeding scheme used to produce the plot.
    """
    tags = [tag for tag in results.keys() if tag != 'EB_test']
    hlt_effs = [results[tag]['HLT_efficiency'] for tag in tags]
    combined_effs = [results[tag]['Combined_efficiency'] for tag in tags]
    
    
    #efficiency_gains = [(combined - hlt) / hlt * 100 for hlt, combined in zip(hlt_effs, combined_effs)]
    efficiency_gains = []
    for hlt, combined in zip(hlt_effs, combined_effs):
        gain = (combined - hlt) / hlt
        if np.isfinite(gain):
            efficiency_gains.append(gain * 100)
        else:
            efficiency_gains.append(999)  # Replace infinities with a large value
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})
    y = np.arange(len(tags))

    ax1.scatter(hlt_effs, y, label='HLT Efficiency', color='cornflowerblue', s=150, alpha=0.5)
    ax1.scatter(combined_effs, y, label='HLT + AD Efficiency', color='mediumvioletred', s=150, alpha=0.5)

    ax1.set_xlabel('Efficiency', fontsize=15)
    ax1.set_title(f'{seed_type} {eff_type}: {bkg_type}, {target_rate}Hz pure rate', fontsize=16)
    ax1.set_yticks(y)
    ax1.set_yticklabels(tags)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.legend()

    ax2.barh(y, efficiency_gains, color='seagreen', edgecolor='k')
    for i, gain in enumerate(efficiency_gains):
        ax2.text(gain + 0.5, y[i], f'{gain:.4f}%', va='center', color='black')

    # Set x axis limit -------------------------------------------
    # Convert to a NumPy array for easier manipulation
    valid_efficiency_gains = np.array(efficiency_gains)
    
    if valid_efficiency_gains.size > 0:
        # Check if 999 exists and find the next-highest value
        if 999 in valid_efficiency_gains:
            max_limit = np.sort(valid_efficiency_gains[valid_efficiency_gains < 999])[-1] + 2
        else:
            max_limit = np.max(valid_efficiency_gains) + 2
    
        ax2.set_xlim(0, max_limit)
    else:
        ax2.set_xlim(0, 2)  # Fallback if no valid values exist
    # -------------------------------------------------------------
    
    ax2.set_xlabel('Efficiency Gain (%)', fontsize=15)
    ax2.set_title('Relative Efficiency Gain', fontsize=15)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




# -----------------------------------------------------------------------------------------
def load_and_preprocess(train_data_scheme: str, pt_normalization_type=None, L1AD_rate=1000, pt_thresholds=[50, 0, 0, 0], comments=None):
    """
    Loads and preprocesses the training and signal data.

    Inputs:
        train_data_scheme: one of the strings defined below. Defines what data is used for training.
        pt_normalization_type: None, or one of the strings defined below. Defines how pt normalization is done.
        L1AD_rate: the pure rate at which the L1AD algo operates at. Used to calculate which events are L1Seeded.
        pt_thresholds: thresholds for which to zero out objects with pt below that value. [jet_threshold, electron_threshold, muon_threshold, photon_threshold].
        Comments: None or string. Any comments about the data / run that are worth noting down in the training documentation file.

    Returns:
        datasets: dict mapping {dataset_tag : dataset_dict} where dataset_dict is a sub-dict containing the data corresponding to that tag.
        data_info: dict containing information about the training data.
    """
    
    # Check arguments
    allowed_train_data_schemes = ['topo2A_train', 'L1noalg_HLTall', 'topo2A_train+L1noalg_HLTall', 'topo2A_train+overlap']
    allowed_norm_types = ['per_event', 'global_division', 'StandardScaler', 'ZeroAwareStandardScaler']
    if (train_data_scheme not in allowed_train_data_schemes):
        raise ValueError(f"Invalid input: train_data_scheme {train_data_scheme}. Must be either None, or one of {allowed_train_data_schemes}")
    if (pt_normalization_type is not None) and (pt_normalization_type not in allowed_norm_types):
        raise ValueError(f"Invalid input: pt_normalization_type {pt_normalization_type}. Must be one of {allowed_norm_types}")

    # -------------------

    # Load data
    #datasets = load_subdicts_from_h5('/eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/ntuples/lam_output')
    #Testing code
    datasets = load_subdicts_from_h5('/eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/ntuples/lam_output/lam_test')

    # -------------------
    
    # Calculate L1AD threshold for target L1AD_rate using the same events kenny used to train L1AD
    L1AD_threshold, L1AD_total_rate = find_threshold(
        scores=datasets['topo2A_train']['topo2A_AD_scores'], 
        weights=datasets['topo2A_train']['weights'], 
        hlt_pass=datasets['topo2A_train']['passL1'], # for pure rate, the event should pass L1AD, but not passL1. 
        target_rate=L1AD_rate, 
        incoming_rate=40e6
    )
    # now recalculate L1Seeded items with the threshold
    for tag, dict in datasets.items():
        dict['L1Seeded'] = (dict['topo2A_AD_scores'] >= L1AD_threshold)
    
    # -------------------
        
    # initialize training scheme
    if train_data_scheme == 'topo2A_train+L1noalg_HLTall':
        datasets = combine_data(datasets, tags_to_combine=['topo2A_train', 'HLT_noalg_eb_L1All'], new_tag='EB_train')

    elif train_data_scheme == 'topo2A_train+overlap':
        
        # keep ony overlap events
        overlap_idxs = datasets['HLT_noalg_eb_L1All']['L1Seeded'] # include events that also pass our L1AD algo
        datasets['HLT_noalg_eb_L1All'] = {key: value[overlap_idxs] for key, value in datasets['HLT_noalg_eb_L1All'].items()}

        # add them to the training data
        datasets = combine_data(datasets, tags_to_combine=['topo2A_train', 'HLT_noalg_eb_L1All'], new_tag='EB_train')

    elif train_data_scheme == 'L1noalg_HLTall':
        datasets['EB_train'] = datasets.pop('HLT_noalg_eb_L1All')

    elif train_data_scheme == 'topo2A_train':
        datasets['EB_train'] = datasets.pop('topo2A_train')

    # now combine the other EB runs into EB_test
    tags_to_combine = [key for key in datasets.keys() if "EB" in key and key != 'EB_train']
    datasets = combine_data(datasets, tags_to_combine=tags_to_combine, new_tag='EB_test')

    # -------------------
    
    # Recalculate the L1AD threshold and L1Seeded events, this time using the newly created 'EB_test' data
    L1AD_threshold, L1AD_total_rate = find_threshold(
        scores=datasets['EB_test']['topo2A_AD_scores'], 
        weights=datasets['EB_test']['weights'], 
        hlt_pass=datasets['EB_test']['passL1'], # for pure rate, the event should pass L1AD, but not passL1. 
        target_rate=L1AD_rate, 
        incoming_rate=40e6
    )
    # now recalculate L1Seeded items with the threshold
    for tag, dict in datasets.items():
        dict['L1Seeded'] = (dict['topo2A_AD_scores'] >= L1AD_threshold)
    
    # ------------------- 
    
    # save raw data
    for tag, data_dict in datasets.items():
        datasets[tag]['raw_HLT_data'] = np.copy(data_dict['HLT_data'])
        datasets[tag]['raw_L1_data'] = np.copy(data_dict['L1_data'])
        datasets[tag]['raw_Offline_data'] = np.copy(data_dict['Offline_data'])

        
    # -------------------
    
    # Change phi --> delta phi between leading jet
    for tag, dict in datasets.items():
        for data_type in ['HLT', 'L1', 'Offline']:
    
            # Create a mask for zeroed-out objects (where pt is zero)
            zeroed_mask = dict[data_type+'_data'][:, :, 0] == 0
    
            # Find phi of the leading jets in each event
            leading_jet_phi = dict[data_type+'_data'][:, 0, 2]
    
            # Duplicate along the third dimension such that it has the right shape
            if data_type == 'HLT' or data_type == 'L1':
                leading_jet_phi = np.tile(leading_jet_phi, (16, 1)).T
            else:
                leading_jet_phi = np.tile(leading_jet_phi, (15, 1)).T
    
            # update dphi for all objects
            dict[data_type+'_data'][:, :, 2] = dphi(phi1=dict[data_type+'_data'][:, :, 2], phi2=leading_jet_phi)
    
            # Reset phi of zeroed-out objects to zero
            dict[data_type+'_data'][zeroed_mask, 2] = 0

    # -------------------
    
    # Multiply the et of the L1 muons by 1000 to get them into the right units.
    for tag, dict in datasets.items():
        if tag == 'random': continue
        for label, data in dict.items():
            if label == 'L1_data':
                data[:, 9:12, 0] *= 1000

    # -------------------

    # Met bias. For some events, MET was not calculated, so was written as -999.
    # We don't want this to impact training, so we map 0 --> 0.001 and -999 --> 0
    for tag, dict in datasets.items():
        for label, data in dict.items():
            if label.endswith('data') and 'Offline' not in label:
    
                # MET: 0 --> 0.001 and -999 --> 0 and nan --> 0
                MET_zeros = (data[:, -1, 0] == 0) # indices where MET=0
                MET_999 = ((data[:, -1, 0] == -999)) # indices where MET=-999 (not calculated)
                MET_nan = np.isnan(data[:, -1, 2])
                data[MET_zeros, -1, 0] = 0.001
                data[MET_999, -1, :] = 0
                data[MET_nan, -1, :] = 0

    # -------------------

    # Zero out jets
    for tag, dict in datasets.items():
        
        HLT_lowpt_mask = dict['HLT_data'][:, :6, 0] < pt_thresholds[0]
        dict['HLT_data'][:, :6][HLT_lowpt_mask] = 0

        Offline_lowpt_mask = dict['Offline_data'][:, :6, 0] < pt_thresholds[0]
        dict['Offline_data'][:, :6][Offline_lowpt_mask] = 0

        L1_lowpt_mask = dict['L1_data'][:, :6, 0] < pt_thresholds[0]
        dict['L1_data'][:, :6][L1_lowpt_mask] = 0

    # Zero out electrons
    for tag, dict in datasets.items():
        
        HLT_lowpt_mask = dict['HLT_data'][:, 6:9, 0] < pt_thresholds[1]
        dict['HLT_data'][:, 6:9][HLT_lowpt_mask] = 0

        Offline_lowpt_mask = dict['Offline_data'][:, 6:9, 0] < pt_thresholds[1]
        dict['Offline_data'][:, 6:9][Offline_lowpt_mask] = 0

        L1_lowpt_mask = dict['L1_data'][:, 6:9, 0] < pt_thresholds[1]
        dict['L1_data'][:, 6:9][L1_lowpt_mask] = 0

    # Zero out muons
    for tag, dict in datasets.items():
        
        HLT_lowpt_mask = dict['HLT_data'][:, 9:12, 0] < pt_thresholds[2]
        dict['HLT_data'][:, 9:12][HLT_lowpt_mask] = 0
        
        Offline_lowpt_mask = dict['Offline_data'][:, 9:12, 0] < pt_thresholds[2]
        dict['Offline_data'][:, 9:12][Offline_lowpt_mask] = 0
        
        L1_lowpt_mask = dict['L1_data'][:, 9:12, 0] < pt_thresholds[2]
        dict['L1_data'][:, 9:12][L1_lowpt_mask] = 0

    # Zero out photons (taus for L1 objects)
    for tag, dict in datasets.items():
        
        HLT_lowpt_mask = dict['HLT_data'][:, 12:15, 0] < pt_thresholds[3]
        dict['HLT_data'][:, 12:15][HLT_lowpt_mask] = 0
        
        Offline_lowpt_mask = dict['Offline_data'][:, 12:15, 0] < pt_thresholds[3]
        dict['Offline_data'][:, 12:15][Offline_lowpt_mask] = 0

        L1_lowpt_mask = dict['L1_data'][:, 12:15, 0] < pt_thresholds[3]
        dict['L1_data'][:, 12:15][L1_lowpt_mask] = 0

    # -------------------
    # pt normalization
    if pt_normalization_type == 'per_event':
        
        # Normalize pt such that sum(pt) = 10 in each event
        for tag, dict in datasets.items():
            for label, data in dict.items():
                if label.endswith('data'):
                    # sum of the pt in each event
                    sum_pt = np.sum(data[:, :, 0], axis=1, keepdims=True)
            
                    # If the sum is 0, set the sum to 1 to avoid division by 0
                    sum_pt[sum_pt == 0] = 1
            
                    # Divide pt by the sum, multiply by 10
                    data[:, :, 0] *= (10/sum_pt)


    if pt_normalization_type == 'global_division':
        # Scale the pts by dividing by the mean of the sample summed pts, and then multiplying by 5 to get the right OOM

        # Calculate the mean over the training data
        HLT_mean_sum_pts = np.mean(np.sum(datasets['EB_train']['HLT_data'][:, :, 0], axis=1))
        Offline_mean_sum_pts = np.mean(np.sum(datasets['EB_train']['Offline_data'][:, :, 0], axis=1))
        L1_mean_sum_pts = np.mean(np.sum(datasets['EB_train']['L1_data'][:, :, 0], axis=1))
        
        # Scale the pts of all the datasets
        for tag, dict in datasets.items():
            dict['HLT_data'][:, :, 0] = 5 * dict['HLT_data'][:, :, 0] / HLT_mean_sum_pts
            dict['Offline_data'][:, :, 0] = 5 * dict['Offline_data'][:, :, 0] / Offline_mean_sum_pts
            dict['L1_data'][:, :, 0] = 5 * dict['L1_data'][:, :, 0] / L1_mean_sum_pts


    # -------------------

    # Flatten ndarrays for use in DNN  

    for tag, dict in datasets.items():
        for label, data in dict.items():
            if label.endswith('data') and 'Offline' not in label:
                datasets[tag][label] = np.reshape(data, newshape=(-1, 48))
            if label.endswith('data') and label == 'Offline_data':  # Check if the label is 'Offline'
                    #print("in offline loop")
                    datasets[tag][label] = np.reshape(data, newshape=(-1, 45))
                   # For other labels, reshape with 48

    # Split the train data into train + val
    indices = np.arange(len(datasets['EB_train']['HLT_data']))
    #print('len_HLT',len(datasets['EB_train']['HLT_data']))
    train_indices, val_indices = train_test_split(indices, test_size=0.15, random_state=0)
    
    datasets['EB_val'] = {key:value[val_indices] for key, value in datasets['EB_train'].items()}
    datasets['EB_train'] = {key:value[train_indices] for key, value in datasets['EB_train'].items()}

    indices_Offline = np.arange(len(datasets['EB_train']['Offline_data']))
    train_indices_Offline, val_indices_Offline = train_test_split(indices_Offline, test_size=0.15, random_state=0)
    #print('len_Offline',len(datasets['EB_train']['Offline_data']))
    datasets['EB_val'] = {key:value[val_indices_Offline] for key, value in datasets['EB_train'].items()}
    datasets['EB_train'] = {key:value[train_indices_Offline] for key, value in datasets['EB_train'].items()}
    

    # -------------------
    if pt_normalization_type.endswith('Scaler'):
        if pt_normalization_type == 'ZeroAwareStandardScaler':
            # Create scalers for HLT and L1 data
            scaler_HLT = ZeroAwareStandardScaler()
            scaler_L1 = ZeroAwareStandardScaler()
        elif pt_normalization_type == 'StandardScaler':
            scaler_HLT = StandardScaler()
            scaler_L1 = StandardScaler()

        
        # Fit the scalers on the training data (EB_train)
        scaler_HLT.fit(datasets['EB_train']['HLT_data'])
        scaler_Offline.fit(datasets['EB_train']['Offline_data'])
        scaler_L1.fit(datasets['EB_train']['L1_data'])
        
        
        # Now, apply the scaler to all datasets
        for tag, data in datasets.items():
            # Apply the fitted HLT scaler
            datasets[tag]['HLT_data'] = scaler_HLT.transform(data['HLT_data'])
            datasets[tag]['Offline_data'] = scaler_Offline.transform(data['Offline_data'])
            # Apply the fitted L1 scaler
            datasets[tag]['L1_data'] = scaler_L1.transform(data['L1_data'])
    
    # -------------------

    data_info = {
        'train_data_scheme': train_data_scheme,
        'pt_normalization_type': pt_normalization_type,
        'L1AD_rate': L1AD_rate,
        'pt_thresholds': pt_thresholds
    }
    if comments is not None:
        data_info['comments'] = comments

    return datasets, data_info



# -----------------------------------------------------------------------------------------
def create_large_AE(input_dim, h_dim_1, h_dim_2, h_dim_3, h_dim_4, latent_dim, l2_reg=0.01, dropout_rate=0):
    
    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(h_dim_1, kernel_regularizer=regularizers.l2(l2_reg))(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(h_dim_2, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_3, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_4, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    z = layers.Dense(latent_dim, kernel_regularizer=regularizers.l2(l2_reg))(x)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)
    encoder = Model(inputs=encoder_inputs, outputs=z)

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(h_dim_4, kernel_regularizer=regularizers.l2(l2_reg))(decoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_3, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(h_dim_2, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(h_dim_1, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(input_dim, kernel_regularizer=regularizers.l2(l2_reg))(x)

    decoder = Model(inputs=decoder_inputs, outputs=outputs)

    ae_outputs = decoder(encoder(encoder_inputs))
    ae = Model(encoder_inputs, outputs=ae_outputs)

    return ae, encoder, decoder
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def create_small_AE(input_dim, h_dim_1, h_dim_2, latent_dim, l2_reg=0.01, dropout_rate=0):
    
    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(h_dim_1, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(h_dim_2, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    z = layers.Dense(latent_dim, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    encoder = Model(inputs=encoder_inputs, outputs=z)

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(h_dim_2, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(decoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(h_dim_1, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(input_dim, kernel_regularizer=regularizers.l2(l2_reg))(x)

    decoder = Model(inputs=decoder_inputs, outputs=outputs)

    ae_outputs = decoder(encoder(encoder_inputs))
    ae = Model(encoder_inputs, outputs=ae_outputs)

    return ae, encoder, decoder


# -----------------------------------------------------------------------------------------
def loss_fn(y_true, y_pred):
    """Masked MSE with correct averaging by the number of valid objects."""
    
    # Masks to filter out invalid objects (zero padding and -999 placeholder)
    mask0 = K.cast(K.not_equal(y_true, 0), K.floatx())
    maskMET = K.cast(K.not_equal(y_true, -999), K.floatx())

    # Mask to upweight the first 6 elements (first two jets)
    weight = 1
    weight_mask = tf.ones_like(y_true)
    weight_mask = tf.concat([tf.ones_like(y_true[:, :6]) * weight, 
                      tf.ones_like(y_true[:, 6:])], 1)
    
    mask = mask0 * maskMET
    
    # Apply the mask to the squared differences
    squared_difference = K.square(mask * (y_pred - y_true)) * weight_mask
    
    # Sum the squared differences and the mask (to count valid objects)
    sum_squared_difference = K.sum(squared_difference, 1)
    valid_count = K.sum(mask, 1)  # Number of valid objects
    
    # Replace 0s by 1s
    valid_count = tf.where(K.equal(valid_count, 0), tf.ones_like(valid_count), valid_count)
    
    # Calculate the mean squared error by dividing by the number of valid objects
    mean_squared_error = sum_squared_difference / valid_count
    
    # Return the mean over the batch
    return K.mean(mean_squared_error)
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def initialize_model(input_dim, dropout_p=0, L2_reg_coupling=0, latent_dim=4, large_network=True, saved_model_path=None, save_version=None):
    '''
    Inputs:
        save_path: string of the path to save the model.
        dropout_p: dropout percentage for the AE.
        L2_reg_coupling: coupling value for L2 regularization.
        latent_dim: dimension of the latent space of the model.
        large_network: boolean for whether the network should be large or small.
        saved_model_path: None or string. If string, loads the weights from the saved model path.
        save_version: None or string. If string, suffix of the model to be loaded.

    Returns:
        HLT_AE: full autoencoder model to be used with HLT objects
        HLT_encoder: just the encoder of HLT_AE
        L1_AE: full autoencoder model to be used with L1 objects
        L1_encoder: just the encoder of L1_AE
    
    '''

    # Initialize models
    if large_network:
        INPUT_DIM = input_dim
        H_DIM_1 = 100
        H_DIM_2 = 100
        H_DIM_3 = 64
        H_DIM_4 = 32
        LATENT_DIM = latent_dim
        
        HLT_AE, HLT_encoder, HLT_decoder = create_large_AE(INPUT_DIM, H_DIM_1, H_DIM_2, H_DIM_3, H_DIM_4, LATENT_DIM, l2_reg=L2_reg_coupling, dropout_rate=dropout_p)    
        L1_AE, L1_encoder, L1_decoder = create_large_AE(INPUT_DIM, H_DIM_1, H_DIM_2, H_DIM_3, H_DIM_4, LATENT_DIM, l2_reg=L2_reg_coupling, dropout_rate=dropout_p)
    else:
        INPUT_DIM = input_dim
        H_DIM_1 = 32
        H_DIM_2 = 8
        LATENT_DIM = latent_dim
        
        HLT_AE, HLT_encoder, HLT_decoder = create_small_AE(INPUT_DIM, H_DIM_1, H_DIM_2, LATENT_DIM, l2_reg=L2_reg_coupling, dropout_rate=dropout_p)
        L1_AE, L1_encoder, L1_decoder = create_small_AE(INPUT_DIM, H_DIM_1, H_DIM_2, LATENT_DIM, l2_reg=L2_reg_coupling, dropout_rate=dropout_p)
    # -------------------

    # Compile
    HLT_AE.compile(optimizer='adam', loss=loss_fn, weighted_metrics=[])
    L1_AE.compile(optimizer='adam', loss=loss_fn, weighted_metrics=[])
    # -------------------

    # Load model weights (if specified in the args)
    if (saved_model_path is None) != (save_version is None):
        raise ValueError("Either both or neither of 'saved_model_path' and 'save_version' should be None.")
        
    if (saved_model_path is not None) and (save_version is not None):
        HLT_AE.load_weights(f'{saved_model_path}/EB_HLT_{save_version}.keras')
        HLT_encoder.load_weights(f'{saved_model_path}/EB_HLT_encoder_{save_version}.keras')
        L1_AE.load_weights(f'{saved_model_path}/EB_L1_{save_version}.keras')
        L1_encoder.load_weights(f'{saved_model_path}/EB_L1_encoder_{save_version}.keras')
    # -------------------

    return HLT_AE, HLT_encoder, L1_AE, L1_encoder

def initialize_model_Offline(input_dim, dropout_p=0, L2_reg_coupling=0, latent_dim=4, large_network=True, saved_model_path=None, save_version=None):
    '''
    Inputs:
        save_path: string of the path to save the model.
        dropout_p: dropout percentage for the AE.
        L2_reg_coupling: coupling value for L2 regularization.
        latent_dim: dimension of the latent space of the model.
        large_network: boolean for whether the network should be large or small.
        saved_model_path: None or string. If string, loads the weights from the saved model path.
        save_version: None or string. If string, suffix of the model to be loaded.

    Returns:
        Offline_AE: full autoencoder model to be used with Offline objects
        Offline_encoder: just the encoder of Offline_AE
    
    '''

    # Initialize models
    if large_network:
        INPUT_DIM = input_dim
        H_DIM_1 = 100
        H_DIM_2 = 100
        H_DIM_3 = 64
        H_DIM_4 = 32
        LATENT_DIM = latent_dim
        
        
        Offline_AE, Offline_encoder, Offline_decoder = create_large_AE(INPUT_DIM, H_DIM_1, H_DIM_2, H_DIM_3, H_DIM_4, LATENT_DIM, l2_reg=L2_reg_coupling, dropout_rate=dropout_p)
    
    else:
        INPUT_DIM = input_dim
        H_DIM_1 = 32
        H_DIM_2 = 8
        LATENT_DIM = latent_dim
                
        Offline_AE, Offline_encoder, Offline_decoder = create_small_AE(INPUT_DIM, H_DIM_1, H_DIM_2, LATENT_DIM, l2_reg=L2_reg_coupling, dropout_rate=dropout_p)
    # -------------------

    # Compile - Here have to use DisCo as the loss_fn
    Offline_AE.compile(optimizer='adam', loss=loss_fn, weighted_metrics=[])
    # -------------------

    # Load model weights (if specified in the args)
    if (saved_model_path is None) != (save_version is None):
        raise ValueError("Either both or neither of 'saved_model_path' and 'save_version' should be None.")
        
    if (saved_model_path is not None) and (save_version is not None):

        Offline_AE.load_weights(f'{saved_model_path}/EB_Offline_{save_version}.keras')
        Offline_encoder.load_weights(f'{saved_model_path}/EB_Offline_encoder_{save_version}.keras')
    # -------------------

    return Offline_AE, Offline_encoder


# -----------------------------------------------------------------------------------------
def train_model(datasets: dict, model_version: str, save_path: str, dropout_p=0, L2_reg_coupling=0, latent_dim=4, large_network=True, training_weights=True):
    """
    Trains, and saves an AE.

    Inputs:
        datasets: dict containing the data.
        
    Returns: None
    """

    model_args = {
        'input_dim': datasets['EB_train']['HLT_data'].shape[1],
        'dropout_p': dropout_p,
        'L2_reg_coupling': L2_reg_coupling,
        'latent_dim': latent_dim,
        'large_network': large_network
    }
    
    model_args_Offline = {
        'input_dim': datasets['EB_train']['Offline_data'].shape[1],
        'dropout_p': dropout_p,
        'L2_reg_coupling': L2_reg_coupling,
        'latent_dim': latent_dim,
        'large_network': large_network
    }
 
    
    HLT_AE, HLT_encoder, L1_AE, L1_encoder = initialize_model(**model_args)
    Offline_AE, Offline_encoder = initialize_model_Offline(**model_args_Offline)

    # Define callbacks
    STOP_PATIENCE = 9
    LR_PATIENCE = 6
    early_stopping = EarlyStopping(patience=STOP_PATIENCE, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=LR_PATIENCE, verbose=1)
    callbacks = [early_stopping, reduce_lr]
    # -------------------

    # Train and save models
    NUM_EPOCHS = 5
    BATCH_SIZE = 512

    #SS - No Offline change here?
    if training_weights:
        train_weights = 1000*datasets['EB_train']['weights']/np.sum(datasets['EB_val']['weights'])
        val_weights = 1000*datasets['EB_val']['weights']/np.sum(datasets['EB_val']['weights'])
    else:
        train_weights = np.ones_like(datasets['EB_train']['weights'])
        val_weights = np.ones_like(datasets['EB_val']['weights'])
    #print('x_HLT', datasets['EB_train']['HLT_data'].shape, datasets['EB_train']['HLT_data'].size)
    #print('y_HLT', datasets['EB_train']['HLT_data'].shape, datasets['EB_train']['HLT_data'].size)
    
    history = HLT_AE.fit(
        x=datasets['EB_train']['HLT_data'], 
        y=datasets['EB_train']['HLT_data'], 
        validation_data=(datasets['EB_val']['HLT_data'], datasets['EB_val']['HLT_data'], val_weights),
        epochs=NUM_EPOCHS, 
        batch_size=BATCH_SIZE, 
        callbacks=callbacks, 
        sample_weight = train_weights,
        verbose=0
    )

    HLT_AE.save_weights(f'{save_path}/EB_HLT_{model_version}.keras')
    HLT_encoder.save_weights(f'{save_path}/EB_HLT_encoder_{model_version}.keras')

    #print('x_Offline', datasets['EB_train']['Offline_data'].shape, datasets['EB_train']['Offline_data'].size)
    #print('y_Offline', datasets['EB_train']['Offline_data'].shape, datasets['EB_train']['Offline_data'].size)

    history = Offline_AE.fit(
        x=datasets['EB_train']['Offline_data'], 
        y=datasets['EB_train']['Offline_data'], 
        validation_data=(datasets['EB_val']['Offline_data'], datasets['EB_val']['Offline_data'], val_weights),
        epochs=NUM_EPOCHS, 
        batch_size=BATCH_SIZE, 
        callbacks=callbacks, 
        sample_weight = train_weights,
        verbose=0
    )

    Offline_AE.save_weights(f'{save_path}/EB_Offline_{model_version}.keras')
    Offline_encoder.save_weights(f'{save_path}/EB_Offline_encoder_{model_version}.keras')

    history = L1_AE.fit(
        x=datasets['EB_train']['L1_data'], 
        y=datasets['EB_train']['L1_data'], 
        validation_data=(datasets['EB_val']['L1_data'], datasets['EB_val']['L1_data'], val_weights),
        epochs=NUM_EPOCHS, 
        batch_size=BATCH_SIZE, 
        callbacks=callbacks, 
        sample_weight = train_weights,
        verbose=0
    )

    L1_AE.save_weights(f'{save_path}/EB_L1_{model_version}.keras')
    L1_encoder.save_weights(f'{save_path}/EB_L1_encoder_{model_version}.keras')
    # -------------------
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def train_multiple_models(datasets: dict, data_info: dict, save_path: str, dropout_p=0, L2_reg_coupling=0, latent_dim=4, large_network=True, num_trainings=20, training_weights=True):
    """
    calls 'initialize_and_train' multiple times to average results across multiple trainings.

    Inputs:
        datasets: dict containing the data.
        data_info: dict containing information about the training data.
        save_path: string of the path to save the model.
        dropout_p: dropout percentage for the AE.
        L2_reg_coupling: coupling value for L2 regularization.
        latent_dim: dimension of the latent space of the model.
        large_network: boolean for whether the network should be large or small.
        num_trainings: number of trainings to run.

    Returns:
        training_info: dict containing information about the training which will be used for processing the model. Also dumped into a file for documentation.
        data_info: dict containing information about the training data.
    """
    
    # -------------------
    print(f'Booting up... initializing trainings of {num_trainings} models\n')

    for i in range(num_trainings):
        print(f'starting training model {i}...')
        
        model_version = f'{i}'
        
        train_model(
            datasets=datasets, 
            model_version=model_version,
            save_path=save_path,
            dropout_p=dropout_p,
            L2_reg_coupling=L2_reg_coupling,
            latent_dim=latent_dim,
            large_network=large_network,
            training_weights=training_weights
        )
        print(f'model {i} success\n')

    print(f'Powering off... finished trainings.')
    # -------------------

    # -------------------
    training_info = {
        'save_path': save_path,
        'dropout_p': dropout_p,
        'L2_reg_coupling': L2_reg_coupling,
        'latent_dim': latent_dim,
        'large_network': large_network,
        'num_trainings': num_trainings,
        'training_weights': training_weights
    }
    # -------------------

    # Write the training info to a txt file
    with open('./training_documentation.txt', 'a') as f:
        f.write('\n')
        f.write(json.dumps(training_info))
        f.write(json.dumps(data_info))
        f.write('\n')
    # -------------------

    return training_info, data_info
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def MSE_AD_score(y, x):
    '''
    Calculates MSE between element of y, x, skipping over missing objects

    Inputs:
        y: array of data (e.g. original samples)
        x: other array of data (e.g. model outputs)

    Returns:
        loss: Array of (masked) MSE values between y and x (e.g. loss[i] = MSE(y, x))
    '''
    # Create the mask where the valid data points are (not 0 or -999)
    mask = (y != 0) & (y != -999)
    
    _x = x * mask
    _y = y * mask
    
    squared_diff = np.square(_y - _x)
    sum_squared_diff = np.sum(squared_diff, axis=-1)
    
    # Count the number of valid (non-masked) elements
    valid_count = np.sum(mask, axis=-1)
    
    # Avoid division by zero by replacing 0 counts with 1
    valid_count = np.where(valid_count == 0, 1, valid_count)
    
    # Calculate the mean squared error for each event
    loss = sum_squared_diff / valid_count
    
    return loss
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def plot_l1Seeded(dataset, TSE_save_path=None, ASE_save_path=None, MiSE_save_path=None, bkg_type='HLT', target_rate=10, L1AD_rate=1000, skipJZ=False):
    '''
    Assumes that all events seen by HLTAD will be seeded from L1AD. 
    Calculates and saves plots of ASE, TSE and MiSE gains (gain over current HLT) for each signal under this scheme. 
    Refer to int note for definitions of ASE, TSE, and MiSE.

    Inputs:
        dataset: dict containing all the data
        ASE_save_path: None or string. if string, path (including filename) in which to save the ASE plot. If none, will not generate the plots.
        TSE_save_path: None or string. if string, path (including filename) in which to save the TSE plot. If none, will not generate the plots.
        MiSE_save_path: None or string. if string, path (including filename) in which to save the MiSE plot. If none, will not generate the plots.
        bkg_type: string, either 'HLT' or 'L1'. Specifies the type of objects used.
        target_rate: target pure rate for HLTAD.
        L1AD_rate: incoming L1AD rate
        skipJZ: bool, True means to not include dijet samples in the plots (since they can mess with the scale).

    Returns:
        ASE_results: dictionary maps {signal_name : sub_dict},
            sub_dict contains ASE value of current HLT as well as ASE value of HLT + AD
        TSE_rsults: dictionary maps {signal_name : sub_dict},
            sub_dict contains TSE value of current HLT as well as TSE value of HLT + AD
        MiSE_rsults: dictionary maps {signal_name : sub_dict},
            sub_dict contains MiSE value of current HLT as well as TSE value of HLT + AD
    '''
    
    if skipJZ:
        skip_tags = 'EB_train', 'jjJZ2', 'jjJZ1', 'jjJZ4', 'EB_val'
    else:
        skip_tags = 'EB_train', 'EB_val'

    # Calculate L1AD threshold for target L1AD_rate
    L1AD_threshold, L1_total_rate = find_threshold(scores=dataset['EB_test']['topo2A_AD_scores'], weights=dataset['EB_test']['weights'], hlt_pass=dataset['EB_test']['passL1'], target_rate=L1AD_rate, incoming_rate=40e6)
    # now recalculate L1Seeded items with the threshold
    for tag, dict in dataset.items():
        dict['L1Seeded2'] = (dict['topo2A_AD_scores'] >= L1AD_threshold)

    # Calculate threshold only considering L1 seeded events
    idxs = dataset['EB_test']['L1Seeded2']
    bkg_scores = dataset['EB_test'][f'{bkg_type}_AD_scores'][idxs]
    bkg_weights = dataset['EB_test'][f'weights'][idxs]
    bkg_hlt_pass = dataset['EB_test']['passHLT'][idxs]
    threshold, total_rate = find_threshold(scores=bkg_scores, weights=bkg_weights, hlt_pass=bkg_hlt_pass, target_rate=10, incoming_rate=L1AD_rate)
    print(f"HLTAD threshold: {threshold}")
    
    
    # ASE (algorithm signal efficiency)
    # Calculate signal ASE values
    ASE_results = {}
    for tag, dict in dataset.items():
        if tag in skip_tags: continue

        # Collect only L1 Seeded events for efficiency calculation
        idxs = dict['L1Seeded2']
        anomalous = (dict[f'{bkg_type}_AD_scores'] >= threshold)[idxs]
        weights = dict['weights'][idxs]
        hlt_pass = dict['passHLT'][idxs]
        hlt_eff, combined_eff = calculate_efficiencies(anomalous, weights, hlt_pass)
        ASE_results[tag] = {
                'HLT_efficiency': hlt_eff,
                'Combined_efficiency': combined_eff
            }

    # TSE (total signal efficiency)
    TSE_results = {}
    for tag, dict in dataset.items():
        if tag in skip_tags: continue

        # Collect all events for efficiency calculation, but only trigger on L1Seeded events
        idxs = dict['L1Seeded2']
        anomalous = (dict[f'{bkg_type}_AD_scores'] >= threshold) & (idxs)
        weights = dict['weights']
        hlt_pass = dict['passHLT']
        hlt_eff, combined_eff = calculate_efficiencies(anomalous, weights, hlt_pass)
        TSE_results[tag] = {
                'HLT_efficiency': hlt_eff,
                'Combined_efficiency': combined_eff
            }

    # MiSE (mixed signal efficiency) with custom efficiency calculation logic
    MiSE_results = {}
    for tag, dict in dataset.items():
        if tag in skip_tags: continue

        # For HLT MiSE, events can pass *any* L1 seed including L1AD
        idxs = (dict['passL1']) | (dict['L1Seeded2'])
        weights = dict['weights'][idxs]
        hlt_pass = dict['passHLT'][idxs]

        # require anomalous events to be L1Seeded
        anomalous = (((dict[f'{bkg_type}_AD_scores'] >= threshold) & (dict['L1Seeded2'])) | dict['passHLT'])[idxs]

        hlt_eff, combined_eff = calculate_efficiencies(anomalous, weights, hlt_pass)

        MiSE_results[tag] = {
                'HLT_efficiency': hlt_eff,
                'Combined_efficiency': combined_eff
            }

    # Generate plots
    if (ASE_save_path is not None) and (TSE_save_path is not None) and (MiSE_save_path is not None):
        plot_efficiencies(ASE_results, bkg_type=bkg_type, save_path=ASE_save_path, target_rate=10, eff_type='ASE', seed_type='l1Seeded')
        plot_efficiencies(TSE_results, bkg_type=bkg_type, save_path=TSE_save_path, target_rate=10, eff_type='TSE', seed_type='l1Seeded')
        plot_efficiencies(MiSE_results, bkg_type=bkg_type, save_path=MiSE_save_path, target_rate=10, eff_type='MiSE', seed_type='l1Seeded')

    return ASE_results, TSE_results, MiSE_results
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def plot_l1All(dataset, TSE_save_path=None, ASE_save_path=None, bkg_type='HLT', target_rate=10, L1AD_rate=1000, plot=False, skipJZ=False):
    '''
    Assumes that HLTAD will see all events passing any L1 seed. 
    Calculates and plots ASE and TSE gains (gain over current HLT) for each signal under this scheme. 
    Refer to int note for definitions of ASE and TSE.

    Inputs:
        dataset: dict containing all the data
        ASE_save_path: None or string. if string, path (including filename) in which to save the ASE plot. If none, will not generate the plots.
        TSE_save_path: None or string. if string, path (including filename) in which to save the TSE plot. If none, will not generate the plots.
        bkg_type: string, either 'HLT' or 'L1'. Specifies the type of objects used.
        target_rate: target pure rate for HLTAD.
        L1AD_rate: incoming L1AD rate
        skipJZ: bool, True means to not include dijet samples in the plots (since they can mess with the scale).

    Returns:
        ASE_results: dictionary maps {signal_name : sub_dict},
            sub_dict contains ASE value of current HLT as well as ASE value of HLT + AD
        TSE_rsults:dictionary maps {signal_name : sub_dict},
            sub_dict contains TSE value of current HLT as well as TSE value of HLT + AD
    '''
    
    if skipJZ:
        skip_tags = 'EB_train', 'jjJZ2', 'jjJZ1', 'jjJZ4', 'EB_val'
    else:
        skip_tags = 'EB_train', 'EB_val'

    # Calculate L1AD threshold for target L1AD_rate
    L1AD_threshold, L1AD_total_rate = find_threshold(scores=dataset['EB_test']['topo2A_AD_scores'], weights=dataset['EB_test']['weights'], hlt_pass=dataset['EB_test']['passL1'], target_rate=L1AD_rate, incoming_rate=40e6)

    # now recalculate L1Seeded items with the threshold
    for tag, dict in dataset.items():
        dict['L1Seeded2'] = (dict['topo2A_AD_scores'] >= L1AD_threshold)

    # Calculate threshold over all passL1 or L1Seeded events
    idxs = (dataset['EB_test']['L1Seeded2']) | (dataset['EB_test']['passL1'])
    bkg_scores = dataset['EB_test'][f'{bkg_type}_AD_scores'][idxs]
    bkg_weights = dataset['EB_test'][f'weights'][idxs]
    bkg_hlt_pass = dataset['EB_test']['passHLT'][idxs]
    threshold, total_rate = find_threshold(scores=bkg_scores, weights=bkg_weights, hlt_pass=bkg_hlt_pass, target_rate=10, incoming_rate=100000)
    
    
    # ASE (algorithm signal efficiency)
    # Calculate signal ASE values
    ASE_results = {}
    for tag, dict in dataset.items():
        if tag in skip_tags: continue

        # Collect either L1 Seeded or passL1 events for efficiency calculation
        idxs = (dict['L1Seeded2']) | (dict['passL1'])
        anomalous = (dict[f'{bkg_type}_AD_scores'] >= threshold)[idxs]
        weights = dict['weights'][idxs]
        hlt_pass = dict['passHLT'][idxs]
        hlt_eff, combined_eff = calculate_efficiencies(anomalous, weights, hlt_pass)
        ASE_results[tag] = {
                'HLT_efficiency': hlt_eff,
                'Combined_efficiency': combined_eff
            }

    # TSE (total signal efficiency)
    TSE_results = {}
    for tag, dict in dataset.items():
        if tag in skip_tags: continue

        # Collect all events for efficiency calculation, but only trigger on L1Seeded or passL1 events
        idxs = (dict['L1Seeded2']) |(dict['passL1'])
        anomalous = (dict[f'{bkg_type}_AD_scores'] >= threshold) & (idxs)
        weights = dict['weights']
        hlt_pass = dict['passHLT']
        hlt_eff, combined_eff = calculate_efficiencies(anomalous, weights, hlt_pass)
        TSE_results[tag] = {
                'HLT_efficiency': hlt_eff,
                'Combined_efficiency': combined_eff
            }
    
        # idxs = (dict['L1Seeded']) |(dict['passL1'])
        # anomalous = (dict[f'{bkg_type}_AD_scores'] >= threshold) & (idxs)
        # weights = dict['weights']
        # hlt_pass = dict['passHLT']
        # hlt_eff, combined_eff = calculate_efficiencies(anomalous, weights, hlt_pass)
        hlt_eff = np.sum(hlt_pass * weights) / np.sum(weights)
        combined_eff = np.sum((hlt_pass | anomalous) * weights) / np.sum(weights)
        # TSE_results[tag] = {
        #         'HLT_efficiency': hlt_eff,
        #         'Combined_efficiency': combined_eff
        #     }

    # Plot
    if (TSE_save_path is not None) and (ASE_save_path is not None):
        plot_efficiencies(ASE_results, bkg_type=bkg_type, save_path=ASE_save_path, target_rate=10, eff_type='ASE', seed_type='l1All')
        plot_efficiencies(TSE_results, bkg_type=bkg_type, save_path=TSE_save_path, target_rate=10, eff_type='TSE', seed_type='l1All')

    return ASE_results, TSE_results
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def l1Seeded_ROC_curve(dataset, save_path=None, model_version=0, obj_type='HLT',  L1AD_rate=1000, target_rate=10):
    '''
    Assumes that all events seen by HLTAD will be seeded from L1AD. 
    Plots ROC curve under this scheme, where TPR = num_triggered / num_passed_L1. 
    Refer to int note for definitions of ASE and TSE.

    Inputs:
        dataset: dict containing all the data
        save_path: None or string. if string, path in which to save the plots. If none, will not generate the plots.
        obj_type: string, either 'HLT' or 'L1'. Specifies the type of objects used.
        target_rate: target pure rate for HLTAD.
        L1AD_rate: incoming L1AD rate

    Returns: 
        signal_efficiencies: dict mapping {signal_name : FPR at the rate corresponding to target_rate}
    '''

    signal_efficiencies = {}
    
    if save_path is not None:
        plt.figure(figsize=(10, 8))
        plt.rcParams['axes.linewidth'] = 2.4
    
    # Calculate the L1 seeded events using the expected L1AD rate
    L1AD_threshold, total_L1AD_rate = find_threshold(scores=dataset['EB_test']['topo2A_AD_scores'], weights=dataset['EB_test']['weights'], hlt_pass=dataset['EB_test']['passL1'], target_rate=L1AD_rate, incoming_rate=40e6)

    # now recalculate L1Seeded items with the threshold
    for tag, dict in dataset.items():
        dict['L1Seeded2'] = (dict['topo2A_AD_scores'] >= L1AD_threshold)

    skip_tags = ['EB_val', 'EB_train']

    # Calculate background AD scores and weights, filtering for events being seeded by L1AD (seeding condition)
    bkg_seed_mask = dataset['EB_test']['L1Seeded2']
    bkg_scores = dataset['EB_test'][f'{obj_type}_AD_scores'][bkg_seed_mask]
    bkg_weights = dataset['EB_test']['weights'][bkg_seed_mask]

    # Calculate the HLT rate needed for 10Hz
    __, total_HLT_rate = find_threshold(scores=bkg_scores, weights=bkg_weights, hlt_pass=dataset['EB_test']['passHLT'][bkg_seed_mask], target_rate=10, incoming_rate=total_L1AD_rate)

    # Calculate target FPR corresponding to 'target_rate' pure rate
    target_FPR = total_HLT_rate / total_L1AD_rate
        
    for tag, data_dict in dataset.items():
        if tag in skip_tags: continue

        # Calculate the signal AD scores and weights, enforcing the seeding condition
        seed_mask = data_dict['L1Seeded2']
        signal_scores = data_dict[f'{obj_type}_AD_scores'][seed_mask]
        signal_weights = data_dict['weights'][seed_mask]
    
        # Combine the background and signal AD scores and weights
        combined_scores = np.concatenate((bkg_scores, signal_scores), axis=0)
        combined_weights = np.concatenate((bkg_weights, signal_weights), axis=0)

        # binary labels: 0 for background, 1 for signal
        combined_labels = np.concatenate((np.zeros_like(bkg_scores), np.ones_like(signal_scores)), axis=0)

        # Call sklearn ROC curve function and calculate AUC (also with sklearn)
        FPRs, TPRs, thresholds = roc_curve(y_true=combined_labels, y_score=combined_scores, sample_weight=combined_weights)
        AUC = auc(FPRs, TPRs)

        # Caluclate the TPR at the target FPR
        closest_index = np.argmin(np.abs(FPRs - target_FPR))
        corresponding_TPR = TPRs[closest_index]
        signal_efficiencies[tag] = corresponding_TPR
        
        # Plot
        if save_path is not None:
            plt.plot(FPRs, TPRs, label=f'{tag}, AUC={AUC:.3f}', linewidth=1.5)

    if save_path is not None:
        # plot diagonal line
        xx = np.linspace(0, 1, 100)
        plt.plot(xx, xx, color='grey', linestyle='dashed')

        # Plot vertical line corresponding to 10Hz HLTAD rate
        target_FPR = total_HLT_rate / total_L1AD_rate
        plt.plot([target_FPR, target_FPR], [0, 1], color='r', linestyle='dashed')

        # Aesthetics
        plt.grid()
        plt.xlabel('FPR', fontsize=14)
        plt.ylabel('TPR', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'L1 Seeded ROC curves: {obj_type} objects, L1AD rate = {L1AD_rate}Hz', fontsize=14)
        plt.legend(fontsize=12, bbox_to_anchor=(1, 0.5), loc='center left')
        plt.savefig(f'{save_path}/{model_version}_l1Seeded_ROC.png')
        plt.close()

    return signal_efficiencies
# -----------------------------------------------------------------------------------------#


# -----------------------------------------------------------------------------------------
def l1All_ROC_curve(dataset, save_path=None, model_version=0, obj_type='HLT',  L1AD_rate=1000, target_rate=10):
    '''
    Assumes that all events passing any L1 seed (inlcuding L1AD) will be seen by HLTAD. 
    Plots ROC curve under this scheme, where TPR = num_triggered / num_passed_L1. 
    Refer to int note for definitions of ASE and TSE.

    Inputs:
        dataset: dict containing all the data
        save_path: None or string. if string, path in which to save the plots. If none, will not generate the plots.
        obj_type: string, either 'HLT' or 'L1'. Specifies the type of objects used.
        target_rate: target pure rate for HLTAD.
        L1AD_rate: incoming L1AD rate

    Returns: 
        signal_efficiencies: dict mapping {signal_name : FPR at the rate corresponding to target_rate}
    '''

    signal_efficiencies = {}
    
    if save_path is not None:
        plt.figure(figsize=(10, 8))
        plt.rcParams['axes.linewidth'] = 2.4
    
    # Calculate the L1 AD threshold for the target rate
    L1AD_threshold, total_L1AD_rate = find_threshold(scores=dataset['EB_test']['topo2A_AD_scores'], weights=dataset['EB_test']['weights'], hlt_pass=dataset['EB_test']['passL1'], target_rate=L1AD_rate, incoming_rate=40e6)

    # now recalculate L1Seeded items with the threshold
    for tag, dict in dataset.items():
        dict['L1Seeded2'] = (dict['topo2A_AD_scores'] >= L1AD_threshold)

    skip_tags = ['EB_val', 'EB_train']

    # Calculate background AD scores and weights, filtering for events being seeded by any L1 seed (seeding condition)
    bkg_seed_mask = (dataset['EB_test']['L1Seeded2'] | dataset['EB_test']['passL1'])
    bkg_scores = dataset['EB_test'][f'{obj_type}_AD_scores'][bkg_seed_mask]
    bkg_weights = dataset['EB_test']['weights'][bkg_seed_mask]

    # Calculate the HLT rate needed for 10Hz
    __, total_HLT_rate = find_threshold(scores=bkg_scores, weights=bkg_weights, hlt_pass=dataset['EB_test']['passHLT'][bkg_seed_mask], target_rate=10, incoming_rate=100000)

    # Calculate target FPR corresponding to 'target_rate' pure rate
    target_FPR = total_HLT_rate / 100000
        
    for tag, data_dict in dataset.items():
        if tag in skip_tags: continue

        # Calculate the signal AD scores and weights, enforcing the seeding condition
        seed_mask = (data_dict['L1Seeded2'] | data_dict['passL1'])
        signal_scores = data_dict[f'{obj_type}_AD_scores'][seed_mask]
        signal_weights = data_dict['weights'][seed_mask]
    
        # Combine the background and signal AD scores and weights
        combined_scores = np.concatenate((bkg_scores, signal_scores), axis=0)
        combined_weights = np.concatenate((bkg_weights, signal_weights), axis=0)

        # binary labels: 0 for background, 1 for signal
        combined_labels = np.concatenate((np.zeros_like(bkg_scores), np.ones_like(signal_scores)), axis=0)

        # Call sklearn ROC curve function and calculate AUC (also with sklearn)
        FPRs, TPRs, thresholds = roc_curve(y_true=combined_labels, y_score=combined_scores, sample_weight=combined_weights)
        AUC = auc(FPRs, TPRs)

        # Caluclate the TPR at the target FPR
        closest_index = np.argmin(np.abs(FPRs - target_FPR))
        corresponding_TPR = TPRs[closest_index]
        signal_efficiencies[tag] = corresponding_TPR
        
        # Plot
        if save_path is not None:
            plt.plot(FPRs, TPRs, label=f'{tag}, AUC={AUC:.3f}', linewidth=1.5)

    if save_path is not None:
        # plot diagonal line
        xx = np.linspace(0, 1, 100)
        plt.plot(xx, xx, color='grey', linestyle='dashed')

        # Plot vertical line corresponding to 10Hz HLTAD rate
        target_FPR = total_HLT_rate / total_L1AD_rate
        plt.plot([target_FPR, target_FPR], [0, 1], color='r', linestyle='dashed')

        # Aesthetics
        plt.grid()
        plt.xlabel('FPR', fontsize=14)
        plt.ylabel('TPR', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'L1 All ROC curves: {obj_type} objects, L1AD rate = {L1AD_rate}Hz', fontsize=14)
        plt.legend(fontsize=12, bbox_to_anchor=(1, 0.5), loc='center left')
        plt.savefig(f'{save_path}/{model_version}_l1All_ROC.png')
        plt.close()

    return signal_efficiencies
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def plot_l1Seeded_pileup_efficiency(dataset, save_path=None, model_version=None, obj_type='HLT',  L1AD_rate=1000, target_rate=10):
    '''
    to write
    '''
    
    # Calculate the L1 seeded events using the expected L1AD rate
    L1AD_threshold, total_L1AD_rate = find_threshold(scores=dataset['EB_test']['topo2A_AD_scores'], weights=dataset['EB_test']['weights'], hlt_pass=dataset['EB_test']['passL1'], target_rate=L1AD_rate, incoming_rate=40e6)

    # now recalculate L1Seeded items with the threshold
    for tag, dict in dataset.items():
        dict['L1Seeded2'] = (dict['topo2A_AD_scores'] >= L1AD_threshold)


    # Calculate HLT threshold only considering L1 seeded events
    idxs = dataset['EB_test']['L1Seeded2']
    bkg_scores = dataset['EB_test'][f'{obj_type}_AD_scores'][idxs]
    bkg_weights = dataset['EB_test'][f'weights'][idxs]
    bkg_hlt_pass = dataset['EB_test']['passHLT'][idxs]
    HLTAD_threshold, total_HLT_rate = find_threshold(scores=bkg_scores, weights=bkg_weights, hlt_pass=bkg_hlt_pass, target_rate=10, incoming_rate=L1AD_rate)

    # Require anomalous events to be L1Seeded
    anomalous = (dataset['EB_test'][f'{obj_type}_AD_scores'] >= HLTAD_threshold)[idxs]
    pileups = dataset['EB_test']['pileups'][idxs]
    weights = dataset['EB_test']['weights'][idxs]

    # Define pileup bins
    bins = np.linspace(np.min(pileups[pileups != 0])-5, np.max(pileups)+5, 35)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # To plot the bin centers

    # Initialize TEfficiency
    h_total = ROOT.TH1F("h_total", "Total Events", len(bins)-1, bins)
    h_pass = ROOT.TH1F("h_pass", "Passed Events", len(bins)-1, bins)

    # Fill histograms using pileup and weights
    for i in range(len(pileups)):
        h_total.Fill(pileups[i], weights[i])
        if anomalous[i]:
            h_pass.Fill(pileups[i], weights[i])

    # Create TEfficiency object
    eff = ROOT.TEfficiency(h_pass, h_total)

    # Plot efficiency vs pileup using ROOT
    if save_path is not None:
        c = ROOT.TCanvas("c", "Efficiency vs Pileup", 800, 600)
        eff.SetTitle(f"Anomalous Event Efficiency vs Pileup;Pileup;Efficiency")
        eff.Draw("AP")  # A: Axis, P: Points with error bars
        c.SaveAs(f'{save_path}/{model_version}_pileup_efficiency.png')
    
    # # Calculate efficiency using np.histogram
    # total_in_bins, _ = np.histogram(pileups, bins=bins, weights=weights)
    # anomalous_in_bins, _ = np.histogram(pileups[anomalous], bins=bins, weights=weights[anomalous])
    
    # # Avoid division by zero
    # efficiency = np.divide(anomalous_in_bins, total_in_bins, out=np.zeros_like(anomalous_in_bins), where=total_in_bins != 0)
    
    # # Plot the efficiency
    # if save_path is not None:
    #     plt.figure(figsize=(10, 8))
    #     plt.rcParams['axes.linewidth'] = 2.4
    #     plt.plot(bin_centers, efficiency, marker='o', linestyle='-', color='b')
    #     plt.xlabel('Pileup')
    #     plt.ylabel('Efficiency')
    #     plt.title('Anomalous Event Efficiency vs Pileup')
    #     plt.grid(True)
    #     plt.savefig(f'{save_path}/{model_version}_pileup_efficiency_matplotlib.png')
    #     plt.close()
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def plot_efficiency_gain_distribution(results, bkg_type, scheme, save_path, jz=True, L1AD_rate=1000, target_rate=10):
    '''
    Plots a box plot of the efficiency gains. (the distribution is over the number of models that were trained).

    Inputs:
        results: results dict. Will be generated in the process_multiple_models function.
        bkg_type: either 'HLT' or 'L1', defining which kind of objects were used.
        scheme: string describing the signal efficiency scheme used in generating the efficiency gains (e.g. l1SeededASE)
        save_path: path to the dir to save the plot
        jz: bool. If False, doesn't include dijet values, which can often ruin the scale of the plots.
    '''
    
    if jz==False:
        skip_tags = ['EB_test', 'jjJZ1', 'jjJZ2']
        jz_str = 'withoutJZ'
    else:
        skip_tags = ['EB_test']
        jz_str=''

        
    gains = []
    labels = []
    for tag in results[bkg_type][scheme][0].keys():
        if tag in skip_tags: continue
            
        hlt_effs = [model[tag]['HLT_efficiency'] for model in results[bkg_type][scheme]]
        combined_effs = [model[tag]['Combined_efficiency'] for model in results[bkg_type][scheme]]
        gain = [(combined - hlt) / hlt * 100 for hlt, combined in zip(hlt_effs, combined_effs)]
        gains.append(gain)
        labels.append(tag)
    plt.figure(figsize=(12, 6))
    plt.boxplot(gains, labels=labels)
    plt.title(f'Distribution of Efficiency Gains ({bkg_type}, {scheme})')
    plt.ylabel('Efficiency Gain (%)')
    plt.xlabel('Signal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}/efficiency_gain_distribution_{bkg_type}_{scheme}_L1AD_rate{L1AD_rate}_{jz_str}.png')
    plt.close()
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def plot_efficiency_distribution(results, bkg_type, save_path, jz=True, L1AD_rate=1000, target_rate=10, seed_type='l1Seeded'):
    '''
    Plots a box plot of the efficiency gains. (the distribution is over the number of models that were trained).

    Inputs:
        results: results dict. Will be generated in the process_multiple_models function.
        bkg_type: either 'HLT' or 'L1', defining which kind of objects were used.
        save_path: path to the dir to save the plot
        jz: bool. If False, doesn't include dijet values, which can often ruin the scale of the plots.
        seed_type: string, either l1Seeded or l1All, specifies which seeding scheme was used for the signal efficiencies.
    '''
    
    if jz==False:
        skip_tags = ['EB_test', 'jjJZ1', 'jjJZ2']
        jz_str = 'withoutJZ'
    else:
        skip_tags = ['EB_test']
        jz_str=''

    effs = []
    labels = []
    for tag in results[f'{bkg_type}_{seed_type}'][0].keys():
        if tag in skip_tags: continue
            
        eff = [sub_dict[tag] for sub_dict in results[f'{bkg_type}_{seed_type}']]
        effs.append(eff)
        labels.append(tag)

    plt.figure(figsize=(12, 6))
    plt.boxplot(effs, labels=labels)
    plt.title(f'Distribution of Efficiencies ({bkg_type})')
    plt.ylabel('Efficiency')
    plt.xlabel('Signal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{seed_type}_efficiency_distribution_{bkg_type}_L1AD_rate{L1AD_rate}_{jz_str}.png')
    plt.close()
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def process_multiple_models(training_info: dict, data_info: dict, plots_path: str, target_rate: int = 10, L1AD_rate: int = 1000, custom_datasets = None):

    print(f'powering on... preparing to run evals')

    if custom_datasets is not None:
        datasets = custom_datasets
    else:
        # Load data according to the training info
        datasets, data_info = load_and_preprocess(**data_info)
    #print('datasets IN PROCESS_MULTIPLE_MODELS', datasets)

    # Unpack training info
    save_path = training_info['save_path']
    dropout_p = training_info['dropout_p']
    L2_reg_coupling = training_info['L2_reg_coupling']
    latent_dim = training_info['latent_dim']
    large_network = training_info['large_network']
    num_trainings = training_info['num_trainings']

    print(f'evals phase 1 of 2 initiated.')

    # Efficiency gains
    EG_results = {
        'HLT': {'l1SeededASE': [], 'l1SeededTSE': [], 'l1SeededMiSE': [], 'l1AllASE': [], 'l1AllTSE': []},
        'L1': {'l1SeededASE': [], 'l1SeededTSE': [], 'l1SeededMiSE': [], 'l1AllASE': [], 'l1AllTSE': []}
    }                                         

    # Efficiencies
    E_results = {
        'HLT_l1Seeded': [],
        'L1_l1Seeded': [],
        'HLT_l1All': [],
        'L1_l1All': []
    }

    # Loop over each trained model
    output_dir = 'output_files_DisCo'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'ad_scores_noDisCo.h5')
    
    with h5py.File(output_file, 'w') as f:
        for i in range(num_trainings):

            print(f'phase 1: starting evals of model {i}...')

            # Load the model
            HLT_AE, HLT_encoder, L1_AE, L1_encoder = initialize_model(
                input_dim=datasets['EB_train']['HLT_data'].shape[1],
                dropout_p=dropout_p,
                L2_reg_coupling=L2_reg_coupling,
                latent_dim=latent_dim,
                large_network=large_network,
                saved_model_path=save_path,
                save_version=i
            )
        
        #Load Offline model
            Offline_AE, Offline_encoder = initialize_model_Offline(
                input_dim=datasets['EB_train']['Offline_data'].shape[1],
                dropout_p=dropout_p,
                L2_reg_coupling=L2_reg_coupling,
                latent_dim=latent_dim,
                large_network=large_network,
                saved_model_path=save_path,
                save_version=i
            )

            # Pass the data through the model
            skip_tags = ['EB_train', 'EB_val']
            for tag, dict in datasets.items():
                if tag in skip_tags: continue
        
                dict['HLT_model_outputs'] = HLT_AE.predict(dict['HLT_data'], verbose=0)
                #dict['HLT_latent_reps'] = HLT_encoder.predict(dict['HLT_data'])
                dict['Offline_model_outputs'] = Offline_AE.predict(dict['Offline_data'], verbose=1)
                #dict['HLT_latent_reps'] = HLT_encoder.predict(dict['HLT_data'])
                dict['L1_model_outputs'] = L1_AE.predict(dict['L1_data'], verbose=0)
            
                #print('HLT_outputs',dict['HLT_model_outputs'][1:5])
                #print('Offline_outputs',dict['Offline_model_outputs'][1:5])
            
                #dict['L1_latent_reps'] = L1_encoder.predict(dict['L1_data'])

        # Calculate the AD scores
        
            for tag, dict in datasets.items():
                if tag in skip_tags: continue
                
                #for key in dict.keys():
                    #print(key)
                
                #print('tag, dict', tag, datasets.items())
        
                dict['HLT_AD_scores'] = MSE_AD_score(dict['HLT_data'], dict['HLT_model_outputs'])
                dict['Offline_AD_scores'] = MSE_AD_score(dict['Offline_data'], dict['Offline_model_outputs'])
                dict['L1_AD_scores'] = MSE_AD_score(dict['L1_data'], dict['L1_model_outputs'])
            
                #print('Offline_AD_scores',dict['Offline_AD_scores'][1:5])
                #print('HLT_AD_scores',dict['HLT_AD_scores'][1:5])
       
                group = f.create_group(tag)
                group.create_dataset('HLT_AD_scores', data=dict['HLT_AD_scores'])
                group.create_dataset('Offline_AD_scores', data=dict['Offline_AD_scores'])
                group.create_dataset('L1_AD_scores', data=dict['L1_AD_scores'])
                

                 # Create a group for each tag
                for key in ['HLT_data', 'L1_data', 'Offline_data', 'event_numbers', 'passHLT', 'passL1', 
                    'pileups', 'run_numbers', 'topo2A_AD_scores', 'weights', 'L1Seeded', 
                    'raw_HLT_data', 'raw_L1_data', 'raw_Offline_data', 'HLT_model_outputs', 
                    'Offline_model_outputs', 'L1_model_outputs']:
                    if key in dict:
                        #print(f"Saving key '{key}' for tag '{tag}'")
                        group.create_dataset(key, data=dict[key])
                    else:
                        print(f"Key '{key}' not found for tag '{tag}'")
                
                #weights_EBtest=datasets['EB_test']['weights']
                #print('EB_test_WEIGHTS', weights_EBtest[1:5])
                #weights_HLTnoalg=datasets['HLT_noalg_eb_L1All']['weights']
                #print('HLT_noalg_eb_L1All_WEIGHTS', weights_HLTnoalg[1:5])
                
                
        print(f"AD scores saved to {output_file}")
        # Run evals
        
        for bkg_type in ['HLT', 'L1']:
            
            l1seededASE, l1seededTSE, l1seededMiSE = plot_l1Seeded(
                datasets, 
                TSE_save_path=f'{plots_path}/{bkg_type}_l1seeded_TSE_{i}.png',
                ASE_save_path=f'{plots_path}/{bkg_type}_l1seeded_ASE_{i}.png',
                MiSE_save_path=f'{plots_path}/{bkg_type}_l1seeded_MiSE_{i}.png',
                bkg_type=bkg_type, 
                target_rate=target_rate, 
                L1AD_rate=L1AD_rate,
                skipJZ=True
            )
            l1allASE, l1allTSE = plot_l1All(
                datasets, 
                TSE_save_path=f'{plots_path}/{bkg_type}_l1All_TSE_{i}.png',
                ASE_save_path=f'{plots_path}/{bkg_type}_l1All_ASE_{i}.png',
                bkg_type=bkg_type, 
                target_rate=target_rate, 
                L1AD_rate=L1AD_rate,
                skipJZ=True
            )

            EG_results[bkg_type]['l1SeededASE'].append(l1seededASE)
            EG_results[bkg_type]['l1SeededTSE'].append(l1seededTSE)
            EG_results[bkg_type]['l1SeededMiSE'].append(l1seededMiSE)
            EG_results[bkg_type]['l1AllASE'].append(l1allASE)
            EG_results[bkg_type]['l1AllTSE'].append(l1allTSE)

            l1Seeded_signal_efficiencies = l1Seeded_ROC_curve(datasets, save_path=plots_path, model_version=f'{bkg_type}_{i}', obj_type=bkg_type, L1AD_rate=L1AD_rate, target_rate=target_rate)
            E_results[f'{bkg_type}_l1Seeded'].append(l1Seeded_signal_efficiencies)
            
            l1All_signal_efficiencies = l1All_ROC_curve(datasets, save_path=plots_path, model_version=f'{bkg_type}_{i}', obj_type=bkg_type, L1AD_rate=L1AD_rate, target_rate=target_rate)
            E_results[f'{bkg_type}_l1All'].append(l1All_signal_efficiencies)

            plot_l1Seeded_pileup_efficiency(datasets, save_path=plots_path, model_version=f'{bkg_type}_{i}', obj_type=bkg_type,  L1AD_rate=L1AD_rate, target_rate=target_rate)

    
    
    print(f'evals phase 1 complete.')
    print(f'evals phase 2 of 2 initiated.')

    # Save results to file
    with open(f'{plots_path}/stability_EG_results.json', 'w') as f:
        json.dump(EG_results, f)

    with open(f'{plots_path}/stability_E_results.json', 'w') as f:
        json.dump(E_results, f)

    for bkg_type in ['HLT', 'L1']:
        for seed_type in ['l1Seeded', 'l1All']:
            plot_efficiency_distribution(E_results, bkg_type, save_path=plots_path, jz=False, L1AD_rate=L1AD_rate, target_rate=target_rate, seed_type=seed_type)
        
        for scheme in ['l1SeededASE', 'l1SeededTSE', 'l1SeededMiSE', 'l1AllASE', 'l1AllTSE']:
            plot_efficiency_gain_distribution(EG_results, bkg_type, scheme, save_path=plots_path, jz=False, L1AD_rate=L1AD_rate, target_rate=target_rate)

    print(f'evals phase 2 complete, powering down...')
    print(f'goodbye.')
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def save_subdicts_to_h5(main_dict, save_dir):
    """
    Saves each sub-dictionary of NumPy arrays in the main_dict to separate HDF5 files.
    
    Args:
        main_dict (dict): A dictionary of dictionaries where the innermost values are NumPy arrays.
        save_dir (str): The directory where the HDF5 files will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    for sub_dict_name, sub_dict in main_dict.items():
        file_path = os.path.join(save_dir, f"{sub_dict_name}.h5")
        with h5py.File(file_path, 'w') as f:
            for key, arr in sub_dict.items():
                f.create_dataset(key, data=arr)
        print(f"Saved {sub_dict_name} to {file_path}")
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def load_subdicts_from_h5(save_dir):
    """
    Loads sub-dictionaries of NumPy arrays from HDF5 files in a directory and reconstructs the original structure.
    
    Args:
        save_dir (str): The directory where the HDF5 files are stored.
    
    Returns:
        main_dict (dict): A dictionary of dictionaries where the innermost values are NumPy arrays.
    """
    main_dict = {}
    
    for filename in os.listdir(save_dir):
        if filename.endswith(".h5"):
            sub_dict_name = os.path.splitext(filename)[0]
            #print('sub_dict_name', sub_dict_name)
            file_path = os.path.join(save_dir, filename)
            with h5py.File(file_path, 'r') as f:
                sub_dict = {key: np.array(f[key]) for key in f}
                #print('sub_dict', sub_dict)
            main_dict[sub_dict_name] = sub_dict
            print(f"Loaded {sub_dict_name} from {file_path}")
    
    return main_dict
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def save_datasets_with_AD_scores(datasets: dict, training_info: dict, save_dir: str):

    # Unpack training info
    save_path = training_info['save_path']
    dropout_p = training_info['dropout_p']
    L2_reg_coupling = training_info['L2_reg_coupling']
    latent_dim = training_info['latent_dim']
    large_network = training_info['large_network']
    num_trainings = training_info['num_trainings']

    # Load the model
    HLT_AE, HLT_encoder, L1_AE, L1_encoder = initialize_model(
        input_dim=datasets['EB_train']['HLT_data'].shape[1],
        dropout_p=dropout_p,
        L2_reg_coupling=L2_reg_coupling,
        latent_dim=latent_dim,
        large_network=large_network,
        saved_model_path=save_path,
        save_version=0
    )

    # Pass the data through the model
    skip_tags = ['EB_train', 'EB_val']
    for tag, dict in datasets.items():
        if tag in skip_tags: continue
    
        dict['HLT_model_outputs'] = HLT_AE.predict(dict['HLT_data'], verbose=0)
        dict['HLT_latent_reps'] = HLT_encoder.predict(dict['HLT_data'], verbose=0)
        dict['L1_model_outputs'] = L1_AE.predict(dict['L1_data'], verbose=0)
        dict['L1_latent_reps'] = L1_encoder.predict(dict['L1_data'], verbose=0)

    # Calculate the AD scores
    for tag, dict in datasets.items():
        if tag in skip_tags: continue
    
        dict['HLT_AD_scores'] = MSE_AD_score(dict['HLT_data'], dict['HLT_model_outputs'])
        dict['L1_AD_scores'] = MSE_AD_score(dict['L1_data'], dict['L1_model_outputs'])

    save_subdicts_to_h5(main_dict=datasets, save_dir=save_dir)
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def convert_to_onnx(training_info, model_version, object_type, save_dir, opset=13, input_dim=48):
    """
    Recreates a Keras model from saved weights and converts it to ONNX format.
    
    Inputs:
        training_info: dict that holds the options used to train the models.
        model_version: int between 0 and num_trainings. Defines which model will be converted to ONNX
        object_type: string, either 'HLT' or 'L1'. defines whether the HLT-object version or L1-object version will be converted.
        save_dir: path in which to save the onnx model.
        opset: ONNX opset version to use (default is 13).
        input_dim: number of input features used in the model. default is 48 (this is the amount used in all my current versions).
    
    Returns:
        None
    """
    
    # Unpack training info
    save_path = training_info['save_path']
    dropout_p = training_info['dropout_p']
    L2_reg_coupling = training_info['L2_reg_coupling']
    latent_dim = training_info['latent_dim']
    large_network = training_info['large_network']
    num_trainings = training_info['num_trainings']

    # Load the model
    HLT_AE, HLT_encoder, L1_AE, L1_encoder = initialize_model(
        input_dim=input_dim,
        dropout_p=dropout_p,
        L2_reg_coupling=L2_reg_coupling,
        latent_dim=latent_dim,
        large_network=large_network,
        saved_model_path=save_path,
        save_version=model_version
    )

    if object_type == 'HLT':
        # Convert HLT_AE model to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(HLT_AE, opset=opset)
        
        # Save ONNX model
        onnx_file_path = f"{save_dir}/HLT_AE_{model_version}.onnx"
        onnx.save(onnx_model, onnx_file_path)
        print(f"ONNX HLT_AE model saved to: {onnx_file_path}")

    elif object_type == 'L1':
        # Similarly, convert the L1_AE model to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(L1_AE, opset=opset)
        onnx_file_path = f"{save_dir}/L1_AE_{model_version}.onnx"
        onnx.save(onnx_model, onnx_file_path)
        print(f"ONNX L1_AE model saved to: {onnx_file_path}")
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
def compare_tf_with_onnx(datasets: dict, training_info: dict, model_version, onnx_path):
    """
    Compares the outputs of TensorFlow and ONNX models.

    Inputs:
        datasets: dictionary containing data to pass through the models.
        training_info: dictionary with model training details.
        model_version: the version of the model to load.
        onnx_path: path where the ONNX models are stored.

    Returns:
        datasets: updated datasets with both TensorFlow and ONNX results.
    """
    
    # Unpack training info
    save_path = training_info['save_path']
    dropout_p = training_info['dropout_p']
    L2_reg_coupling = training_info['L2_reg_coupling']
    latent_dim = training_info['latent_dim']
    large_network = training_info['large_network']

    # Load the TensorFlow model
    HLT_AE, HLT_encoder, L1_AE, L1_encoder = initialize_model(
        input_dim=datasets['EB_train']['HLT_data'].shape[1],
        dropout_p=dropout_p,
        L2_reg_coupling=L2_reg_coupling,
        latent_dim=latent_dim,
        large_network=large_network,
        saved_model_path=save_path,
        save_version=model_version
    )

    # Pass the data through the TensorFlow model
    skip_tags = ['EB_train', 'EB_val']
    for tag, dict in datasets.items():
        if tag in skip_tags: continue
    
        dict['HLT_model_outputs'] = HLT_AE.predict(dict['HLT_data'], verbose=0)
        dict['L1_model_outputs'] = L1_AE.predict(dict['L1_data'], verbose=0)

    # Calculate the AD scores using TensorFlow model outputs
    for tag, dict in datasets.items():
        if tag in skip_tags: continue
    
        dict['HLT_AD_scores'] = MSE_AD_score(dict['HLT_data'], dict['HLT_model_outputs'])
        dict['L1_AD_scores'] = MSE_AD_score(dict['L1_data'], dict['L1_model_outputs'])

    # Load the ONNX models
    HLT_onnx_session = rt.InferenceSession(f"{onnx_path}/HLT_AE_{model_version}.onnx")
    L1_onnx_session = rt.InferenceSession(f"{onnx_path}/L1_AE_{model_version}.onnx")

    # Run inference using the ONNX models and store results
    for tag, dict in datasets.items():
        if tag in skip_tags: continue

        # Run inference on HLT data
        onnx_inputs_HLT = {HLT_onnx_session.get_inputs()[0].name: dict['HLT_data'].astype(np.float32)}
        dict['ONNX_HLT_model_outputs'] = HLT_onnx_session.run(None, onnx_inputs_HLT)[0]

        # Run inference on L1 data
        onnx_inputs_L1 = {L1_onnx_session.get_inputs()[0].name: dict['L1_data'].astype(np.float32)}
        dict['ONNX_L1_model_outputs'] = L1_onnx_session.run(None, onnx_inputs_L1)[0]

    # Calculate the AD scores using ONNX model outputs
    for tag, dict in datasets.items():
        if tag in skip_tags: continue
    
        dict['ONNX_HLT_AD_scores'] = MSE_AD_score(dict['HLT_data'], dict['ONNX_HLT_model_outputs'])
        dict['ONNX_L1_AD_scores'] = MSE_AD_score(dict['L1_data'], dict['ONNX_L1_model_outputs'])

    return datasets
# -----------------------------------------------------------------------------------------

def load_and_inference(data_info, training_info, model_version=0):
    datasets, data_info = load_and_preprocess(**data_info)
    
    # Unpack training info
    save_path = training_info['save_path']
    dropout_p = training_info['dropout_p']
    L2_reg_coupling = training_info['L2_reg_coupling']
    latent_dim = training_info['latent_dim']
    large_network = training_info['large_network']
    num_trainings = training_info['num_trainings']

    # Load the model
    HLT_AE, HLT_encoder, L1_AE, L1_encoder = initialize_model(
        input_dim=datasets['EB_train']['HLT_data'].shape[1],
        dropout_p=dropout_p,
        L2_reg_coupling=L2_reg_coupling,
        latent_dim=latent_dim,
        large_network=large_network,
        saved_model_path=save_path,
        save_version=model_version
    )

    # Pass the data through the model
    skip_tags = ['EB_train', 'EB_val']
    for tag, dict in datasets.items():
        if tag in skip_tags: continue
    
        dict['HLT_model_outputs'] = HLT_AE.predict(dict['HLT_data'], verbose=0)
        #dict['HLT_latent_reps'] = HLT_encoder.predict(dict['HLT_data'], verbose=0)
        dict['L1_model_outputs'] = L1_AE.predict(dict['L1_data'], verbose=0)
        #dict['L1_latent_reps'] = L1_encoder.predict(dict['L1_data'], verbose=0)

    # Calculate the AD scores
    for tag, dict in datasets.items():
        if tag in skip_tags: continue
    
        dict['HLT_AD_scores'] = MSE_AD_score(dict['HLT_data'], dict['HLT_model_outputs'])
        dict['L1_AD_scores'] = MSE_AD_score(dict['L1_data'], dict['L1_model_outputs'])

    del datasets['EB_train']
    del datasets['EB_val']
    return datasets

def compare_two_models(data_info_1, training_info_1, data_info_2, training_info_2, plots_path, obj_types=['HLT', 'L1'], model_versions=[0, 0], trigger_schemes=['l1Seeded', 'l1All'], model_names=['model_1', 'model_2'], L1AD_rate=1000, target_rate=10):
    """
    Compares two models by generating plots of leading jet pt distributions and 2D histograms of AD scores.

    This function processes datasets from two models, calculates thresholds, and generates plots to compare
    the performance of the models based on their anomaly detection (AD) scores.

    Args:
        data_info_1 (dict): Information required to load and preprocess the first dataset.
        training_info_1 (dict): Training details for the first model.
        data_info_2 (dict): Information required to load and preprocess the second dataset.
        training_info_2 (dict): Training details for the second model.
        plots_path (str): Directory path where the plots will be saved.
        obj_types (list of str): List of object types for each model, e.g., ['HLT', 'L1'].
        model_versions (list of int): List of model versions to load for each dataset.
        trigger_schemes (list of str): List of trigger schemes, either 'l1Seeded' or 'l1All', for each model.
        model_names (list of str): Names of the models for labeling plots.
        L1AD_rate (int): Target L1 anomaly detection rate.
        target_rate (int): Target rate for the HLT anomaly detection.

    Raises:
        ValueError: If any element in trigger_schemes is not 'l1Seeded' or 'l1All'.

    Generates:
        - Histograms of leading jet pt for each tag from both models.
        - 2D histograms of AD scores for all events, events with passAD_idxs1, and events with passAD_idxs2 for each tag.
    """


    plt.rcParams['axes.linewidth'] = 2

    if not all(scheme in ['l1Seeded', 'l1All'] for scheme in trigger_schemes):
        raise ValueError("Both elements of trigger_schemes must be either 'l1Seeded' or 'l1All'")

    datasets = [load_and_inference(data_info_1, training_info_1, model_versions[0]),
                load_and_inference(data_info_2, training_info_2, model_versions[1])]

    plotting_data = {
        'data1': [],
        'data2': [],
        'weights1': [],
        'weights2': [],
        'tags': [],
        'passAD_idxs1': [],
        'passAD_idxs2': []
    }

    for i, (dataset, obj_type, trigger_scheme) in enumerate(zip(datasets, obj_types, trigger_schemes)):
        # Calculate L1AD threshold
        L1AD_threshold, L1_total_rate = find_threshold(
            scores=dataset['EB_test']['topo2A_AD_scores'], 
            weights=dataset['EB_test']['weights'], 
            hlt_pass=dataset['EB_test']['passL1'], 
            target_rate=L1AD_rate, 
            incoming_rate=40e6
        )

        # Recalculate L1Seeded items with the threshold
        for tag, data_dict in dataset.items():
            data_dict['L1Seeded2'] = (data_dict['topo2A_AD_scores'] >= L1AD_threshold)

        # Determine indices based on the trigger scheme
        if trigger_scheme == 'l1Seeded':
            idxs = dataset['EB_test']['L1Seeded2']
        else:  # 'l1All'
            idxs = dataset['EB_test']['L1Seeded2'] | dataset['EB_test']['passL1']

        # Calculate HLTAD threshold
        HLTAD_threshold, HLT_total_rate = find_threshold(
            scores=dataset['EB_test'][f'{obj_type}_AD_scores'][idxs], 
            weights=dataset['EB_test']['weights'][idxs], 
            hlt_pass=dataset['EB_test']['passHLT'][idxs], 
            target_rate=target_rate, 
            incoming_rate=L1_total_rate
        )

        for tag, data_dict in dataset.items():
            # Calculate indices
            seed_idxs = data_dict['L1Seeded2'] if trigger_scheme == 'l1Seeded' else (data_dict['L1Seeded2'] | data_dict['passL1'])
            passAD_idxs = (data_dict[f'{obj_type}_AD_scores'] >= HLTAD_threshold) & seed_idxs

            # Collect data and weights
            print("TAGS", tags, data_dict)
            data = np.reshape(data_dict[f'raw_{obj_type}_data'], newshape=(-1, 16, 3))[passAD_idxs]
            #After adding offline array
            #data = np.reshape(data_dict[f'raw_{obj_type}_data'], newshape=(-1, 15, 3))[passAD_idxs]
            weights = data_dict['weights'][passAD_idxs]
            
            # Append to plotting data
            plotting_data[f'data{i+1}'].append(data)
            # print for some checks:
            print(f'data{i+1} shape: {data.shape}')
            print(f'tag: {tag}')
            print(f'tag index: {np.where(plotting_data["tags"] == tag)}')
            plotting_data[f'weights{i+1}'].append(weights)
            if tag not in plotting_data['tags']:
                plotting_data['tags'].append(tag)
            plotting_data[f'passAD_idxs{i+1}'].append(passAD_idxs)

    # Plot leading jet pt for each tag from each of the two models
    for j, tag in enumerate(plotting_data['tags']):
        for obj_name, obj_idx in zip(['jet', 'el', 'muon', 'photon'], [0, 6, 9, 12]):
            data1 = plotting_data['data1'][j]
            data2 = plotting_data['data2'][j]
            weights1 = plotting_data['weights1'][j]
            weights2 = plotting_data['weights2'][j]
    
            leading_jet_pt1 = np.clip(data1[:, obj_idx, 0], a_min=0, a_max=2000)
            leading_jet_pt2 = np.clip(data2[:, obj_idx, 0], a_min=0, a_max=2000)
            bins = np.linspace(0, 300, 20)
    
            plt.figure(figsize=(10, 6))
            plt.hist(leading_jet_pt1, bins=bins, weights=weights1, density=True, alpha=0.5, label=f'{model_names[0]}', fill=False, linewidth=3, histtype='step')
            plt.hist(leading_jet_pt2, bins=bins, weights=weights2, density=True, alpha=0.5, label=f'{model_names[1]}', fill=False, linewidth=3, histtype='step')
    
            plt.xlabel('Leading Jet pt')
            plt.ylabel('Density')
            plt.title(f'Leading {obj_name} pt Distribution for {tag}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{plots_path}/{tag}_leading_{obj_name}_pt.png")
            plt.close()

    # Extract AD scores and create 2D histograms for each tag
    for j, tag in enumerate(plotting_data['tags']):
        ad_scores1 = np.clip(datasets[0][tag][f'{obj_types[0]}_AD_scores'], a_min=0, a_max=30)
        ad_scores2 = np.clip(datasets[1][tag][f'{obj_types[1]}_AD_scores'], a_min=0, a_max=30)

        # 1. Over all events for the current tag
        plt.figure(figsize=(10, 6))
        plt.hist2d(ad_scores1, ad_scores2, bins=35, density=True, cmap='viridis')
        plt.colorbar(label='Density')
        plt.xlabel(f'{model_names[0]} AD Scores')
        plt.ylabel(f'{model_names[1]} AD Scores')
        plt.title(f'2D Histogram of AD Scores for All Events - {tag}')
        plt.grid(True)
        plt.savefig(f"{plots_path}/{tag}_all_events_ad_scores.png")
        plt.close()

        # 2. Over events with passAD_idxs1 for the current tag
        plt.figure(figsize=(10, 6))
        plt.hist2d(ad_scores1[plotting_data['passAD_idxs1'][j]], ad_scores2[plotting_data['passAD_idxs1'][j]], bins=20, density=True, cmap='viridis')
        plt.colorbar(label='Density')
        plt.xlabel(f'{model_names[0]} AD Scores')
        plt.ylabel(f'{model_names[1]} AD Scores')
        plt.title(f'2D Histogram of AD Scores for Events with passAD_idxs1 - {tag}')
        plt.grid(True)
        plt.savefig(f"{plots_path}/{tag}_passAD_idxs1_ad_scores.png")
        plt.close()

        # 3. Over events with passAD_idxs2 for the current tag
        plt.figure(figsize=(10, 6))
        plt.hist2d(ad_scores1[plotting_data['passAD_idxs2'][j]], ad_scores2[plotting_data['passAD_idxs2'][j]], bins=20, density=True, cmap='viridis')
        plt.colorbar(label='Density')
        plt.xlabel(f'{model_names[0]} AD Scores')
        plt.ylabel(f'{model_names[1]} AD Scores')
        plt.title(f'2D Histogram of AD Scores for Events with passAD_idxs2 - {tag}')
        plt.grid(True)
        plt.savefig(f"{plots_path}/{tag}_passAD_idxs2_ad_scores.png")
        plt.close()