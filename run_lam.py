import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import tensorflow as tf
import load_and_match as lam

L1AD_rate = 1000
target_rate = 10

# Load and match my data with the topo2A (L1AD) scores. Only needs to be done once ever.
#lam.load_and_match('/eos/home-m/mmcohen/ad_trigger_development/data/loaded_and_matched_ntuples/12-13-2024')
lam.load_and_match('/eos/user/s/ssaha/SWAN_projects/AD_Trigger_1/ntuples/lam_output/')