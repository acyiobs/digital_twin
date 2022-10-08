# %%
import os
import numpy as np
import pandas as pd
import scipy.io as scipyio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import utm
from scipy.io import savemat
from tqdm import tqdm


def xy_from_latlong(lat_long):
    """ Assumes lat and long along row. Returns same row vec/matrix on
    cartesian coords."""
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:,0], lat_long[:,1])
    return np.stack((x,y), axis=1)


# The only input needed is the absolute path to the scenario folder:
# Folder containing units' folders and scenarioX.csv
scenario_folder = r'E:\DeepSense\Scenarios\Scenario1\DEV[95%]'


# Automatically fetch CSV (the only csv in folder)
try:
    csv_file = [f for f in os.listdir(scenario_folder) if f.endswith('csv')][0]
    csv_path = os.path.join(scenario_folder, csv_file)
except:
    raise Exception(f'No csv file inside {scenario_folder}.')

dataframe = pd.read_csv(csv_path)
print(f'Columns: {dataframe.columns.values}')
print(f'Number of Rows: {dataframe.shape[0]}')

#%% Load and display power data

N_BEAMS = 64
n_samples = dataframe.index.stop
pwr_rel_paths = dataframe['unit1_pwr_60ghz'].values
pwrs_array = np.zeros((n_samples, N_BEAMS))

for sample_idx in tqdm(range(n_samples)):
    pwr_abs_path = os.path.join(scenario_folder, pwr_rel_paths[sample_idx])
    pwrs_array[sample_idx] = np.loadtxt(pwr_abs_path)

#%% find best beams
best_beams = np.argmax(pwrs_array, 1) + 1 # starts from 1
center_beam = 28
center_ue = np.where(best_beams==center_beam)[0]

#%%
pos_bs = np.loadtxt(os.path.join(scenario_folder, dataframe['unit1_loc'].values[0]))
pos_bs = np.expand_dims(pos_bs, 0)

# UE positions
pos_rel_paths = dataframe['unit2_loc'].values
pos_ue_array = np.zeros((n_samples, 2)) # 2 = Latitude and Longitude

# Load each individual txt file
for sample_idx in range(n_samples):
    pos_abs_path = os.path.join(scenario_folder, pos_rel_paths[sample_idx])
    pos_ue_array[sample_idx] = np.loadtxt(pos_abs_path)


pos_ue_cart = xy_from_latlong(pos_bs)
pos_bs_cart = xy_from_latlong(pos_ue_array)
pos_diff = pos_ue_cart - pos_bs_cart

#%% k = sum(x_i*y_i) / sum(x_i^2)
center_ue_pos = pos_diff[center_ue, :]
k = np.sum(center_ue_pos[:, 1] * center_ue_pos[:, 0]) / np.sum(center_ue_pos[:, 0] * center_ue_pos[:, 0])
BS_oreientation = np.arctan(k) / np.pi*180
print("BS_oreientation: " + str(BS_oreientation) + " (degree)")