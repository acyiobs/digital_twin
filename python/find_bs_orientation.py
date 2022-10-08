# %%
import os
import numpy as np
import pandas as pd
import scipy.io as scipyio
import matplotlib.pyplot as plt
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
center_ue = np.where(best_beams==34)[0]


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

# pos_diff = pos_diff[:,::-1]

# get distances and angle from the transformed position
dist = np.linalg.norm(pos_diff, axis=1)
plt.scatter(pos_diff[:, 0], pos_diff[:, 1])
plt.scatter(0, 0)
plt.axis('equal')
plt.xlabel('X-coordinates (meter)')
plt.ylabel('Y-coordinates (meter)')
# plt.ylim([-20, 0])
plt.show()

# savemat('ue_relative_pos.mat', {'ue_relative_pos': pos_diff})
# savemat('real_beam_pwr.mat', {'real_beam_pwr': pwrs_array})

# %%
first_grid = pos_diff[pos_diff[:, 1]>-18,:]
second_grid = pos_diff[pos_diff[:, 1]<=-18,:]

first_grid_max_x = first_grid[:, 0].max()
first_grid_min_x = first_grid[:, 0].min()

first_grid_max_y = first_grid[:, 1].max()
first_grid_min_y = first_grid[:, 1].min()

first_grid_x_len = first_grid_max_x - first_grid_min_x
first_grid_y_len = first_grid_max_y - first_grid_min_y

second_grid_max_x = second_grid[:, 0].max()
second_grid_min_x = second_grid[:, 0].min()

second_grid_max_y = second_grid[:, 1].max()
second_grid_min_y = second_grid[:, 1].min()

second_grid_x_len = second_grid_max_x - second_grid_min_x
second_grid_y_len = second_grid_max_y - second_grid_min_y

print('done')
