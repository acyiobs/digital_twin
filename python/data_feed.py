"""
author:Shuaifeng
data:10/8/2022
"""
import os
import numpy as np
import pandas as pd
import torch
import random
from skimage import io
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from tqdm import tqdm
import sklearn


def create_samples(pos_path, beam_pwr_path, rand_state, mode, train_split, num_data_point, portion):
    beam_pwr = loadmat(beam_pwr_path)[beam_pwr_path.split('.')[0]]
    ue_relative_pos = loadmat(pos_path)[pos_path.split('.')[0]]
    beam_pwr = beam_pwr[:, 1::4]
    best_beams = np.argmax(beam_pwr, 1) # starts from 0

    polar_anlge = np.arctan2(ue_relative_pos[:, 1], ue_relative_pos[:, 0]) / np.pi
    polar_distance = np.sqrt(ue_relative_pos[:, 1]**2 + ue_relative_pos[:, 0]**2) / np.sqrt(24**2+28**2)

    ue_relative_pos = ue_relative_pos / 30.

    ue_relative_pos = np.concatenate([ue_relative_pos, np.stack([polar_anlge, polar_distance], -1)], -1)

    (beam_pwr, ue_relative_pos, best_beams) = sklearn.utils.shuffle(beam_pwr, ue_relative_pos, best_beams, random_state=rand_state)
    
    num_data = best_beams.size
    num_data = int(num_data * portion)
    beam_pwr, ue_relative_pos, best_beams = beam_pwr[:num_data, ...], ue_relative_pos[:num_data, ...], best_beams[:num_data, ...]

    
    num_data = best_beams.size
    num_data = int(num_data * train_split)
    if mode=='train':
        ue_relative_pos = ue_relative_pos[:num_data, ...]
        best_beams = best_beams[:num_data, ...]
        beam_pwr = beam_pwr[:num_data, ...]
    else:
        ue_relative_pos = ue_relative_pos[num_data:, ...]
        best_beams = best_beams[num_data:, ...]
        beam_pwr = beam_pwr[num_data:, ...]
    
    if num_data_point is None:
        num_data_point = best_beams.size
    if best_beams.size < num_data_point:
        raise Exception("Not enough data point!")

    return ue_relative_pos[:num_data_point, ...], best_beams[:num_data_point, ...], beam_pwr[:num_data_point, ...]


class DataFeed(Dataset):
    def __init__(self, pos_path, beam_pwr_path, rand_state, mode='train', train_split=0.8, num_data_point=None, portion=1.):
        self.ue_relative_pos, self.best_beams, self.beam_pwr = create_samples(pos_path, beam_pwr_path, rand_state, mode, train_split, num_data_point, portion)
    
    def __len__(self):
        return self.best_beams.size
    
    def __getitem__(self, idx):
        ue_relative_pos = self.ue_relative_pos[idx, ...]
        best_beams = self.best_beams[idx, ...]
        beam_pwr = self.beam_pwr[idx, ...]

        ue_relative_pos = torch.tensor(ue_relative_pos, requires_grad=False)
        best_beams = torch.tensor(best_beams, requires_grad=False)
        beam_pwr = torch.tensor(beam_pwr, requires_grad=False)
        
        return ue_relative_pos.float(), best_beams.long(), beam_pwr.float()


if __name__ == "__main__":
    batch_size = 32
    val_batch_size =128

    real_beam_pwr_path = 'real_beam_pwr.mat'
    real_pos_path = 'ue_relative_pos.mat'

    synth_beam_pwr_path = 'synth_beam_power.mat'
    synth_pos_path = 'synth_UE_loc.mat'

    rand_state = 10

    dataset_real_train = DataFeed(pos_path=real_pos_path, beam_pwr_path=real_beam_pwr_path, rand_state=rand_state, mode='train', num_data_point=1928)
    dataset_real_test = DataFeed(pos_path=real_pos_path, beam_pwr_path=real_beam_pwr_path, rand_state=rand_state, mode='test')
    dataset_synth_train = DataFeed(pos_path=synth_pos_path, beam_pwr_path=synth_beam_pwr_path, rand_state=rand_state, mode='train', num_data_point=1928)
    dataset_synth_test = DataFeed(pos_path=synth_pos_path, beam_pwr_path=synth_beam_pwr_path, rand_state=rand_state, mode='test')

    real_train_loader = DataLoader(dataset_real_train, batch_size=batch_size, shuffle=True)
    real_test_loader = DataLoader(dataset_real_test, val_batch_size, shuffle=False)
    synth_train_loader = DataLoader(dataset_synth_train, batch_size=batch_size, shuffle=True)
    synth_test_loader = DataLoader(dataset_synth_test, val_batch_size, shuffle=False)

    (ue_relative_pos, best_beams, beam_pwr) = next(iter(real_train_loader))
    (ue_relative_pos, best_beams, beam_pwr) = next(iter(real_test_loader))
    (ue_relative_pos, best_beams, beam_pwr) = next(iter(synth_train_loader))
    (ue_relative_pos, best_beams, beam_pwr) = next(iter(synth_test_loader))
    
    print('done')

