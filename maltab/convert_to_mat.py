import imp
import numpy as np
from scipy.io import savemat

dataset = np.load("datasets_1.npy", allow_pickle=True, encoding = 'latin1')


pwr_A = np.zeros((6, 41, 16, 16))
pwr_B = np.zeros((6, 41, 16, 16))

for i in range(dataset.shape[0]):
    for j in range(dataset.shape[1]):
        pwr_A[i, j, ...] = dataset[i, j, 0]['pwr_A']
        pwr_B[i, j, ...] = dataset[i, j, 1]['pwr_B']

savemat("dataset_1.mat", {"pwr_A": pwr_A, "pwr_B": pwr_B})

pass