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


def create_samples(mode, train_split, num_data_point, portion):
    beam_pwr = loadmat('synth_beam_power.mat')['synth_beam_power']
    ue_relative_pos = loadmat('synth_UE_loc.mat')['synth_UE_loc']
    best_beams = np.argmax(beam_pwr, 1) # starts from 1
    # ue_gps_pos = loadmat('ue_gps_pos.mat')['ue_gps_pos']

    # ue_relative_pos = ue_gps_pos

    # ue_relative_pos[:, 0] = (ue_relative_pos[:, 0] - ue_relative_pos[:, 0].min()) / (ue_relative_pos[:, 0].max() - ue_relative_pos[:, 0].min())
    # ue_relative_pos[:, 1] = (ue_relative_pos[:, 1] - ue_relative_pos[:, 1].min()) / (ue_relative_pos[:, 1].max() - ue_relative_pos[:, 1].min())

    tmp1 = np.arctan2(ue_relative_pos[:, 1], ue_relative_pos[:, 0]) / np.pi
    tmp2 = np.sqrt(ue_relative_pos[:, 1]**2 + ue_relative_pos[:, 0]**2) / np.sqrt(24**2+28**2)

    ue_relative_pos = ue_relative_pos / 30.

    ue_relative_pos = np.concatenate([ue_relative_pos, np.stack([tmp1, tmp2], -1)], -1)

    (beam_pwr, ue_relative_pos, best_beams) = sklearn.utils.shuffle(beam_pwr, ue_relative_pos, best_beams, random_state=1115)
    
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
    
    if not num_data_point:
        num_data_point = best_beams.size
    if best_beams.size < num_data_point:
        raise Exception("Not enough data point!")

    return ue_relative_pos[:num_data_point, ...], best_beams[:num_data_point, ...], beam_pwr[:num_data_point, ...]


class DataFeed(Dataset):
    def __init__(self, mode='train', train_split=0.8, num_data_point=None, portion=1.):
        self.ue_relative_pos, self.best_beams, self.beam_pwr = create_samples(mode, train_split, num_data_point, portion)
    
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
    train_loader = DataLoader(DataFeed(mode='train', num_data_point=2411), batch_size=32, shuffle=True)
    val_loader = DataLoader(DataFeed(mode='test'), batch_size=32, shuffle=False)
    (ue_relative_pos, best_beams, beam_pwr) = next(iter(train_loader))
    (ue_relative_pos, best_beams, beam_pwr) = next(iter(val_loader))
    
    print('done')
