"""
author:Shuaifeng
data:10/8/2022
"""
import torch
import numpy as np
import datetime
from scipy.io import savemat
from torch.utils.data import DataLoader
from train_model import train_model
import data_feed_real
import data_feed_synth


def train_model_synth(
    num_epoch=80,
    if_writer=False,
):
    test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, PATH = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        comment=comment,
        num_classes=16,
        num_epoch=num_epoch,
        if_writer=if_writer,
    )

    return test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, PATH


if __name__ == "__main__":
    torch.manual_seed(2022)
    now = datetime.datetime.now().strftime("%H_%M_%S")
    date = datetime.date.today().strftime("%y_%m_%d")
    comment = "synth" + now + "_" + date
    num_epoch = 80
    batch_size = 32
    val_batch_size =128

    all_acc = []
    all_pwr = []

    max_data_point = 1928
    for percentage in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        num_data_point = int(max_data_point * percentage)
    
        train_loader = DataLoader(data_feed_real.DataFeed(mode='train', num_data_point=num_data_point), batch_size, shuffle=True)
        val_loader = DataLoader(data_feed_real.DataFeed(mode='test'), val_batch_size, shuffle=False)
        test_loader = DataLoader(data_feed_real.DataFeed(mode='test'), val_batch_size, shuffle=False)

        print('percentage: '+ str(percentage))
        test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, PATH = train_model_synth(num_epoch, if_writer=False)

        all_acc.append(test_acc)
        all_pwr.append(test_pwr)
        # print(test_loss)
        # print(test_acc)
        # print(test_pwr)
    all_acc = np.stack(all_acc, -1)
    all_pwr = np.stack(all_pwr, -1)

    print(all_acc)
    print(all_pwr)
    savemat('all_acc_train_on_real.mat', {'all_acc_train_on_real': all_acc})
    savemat('all_pwr_train_on_real.mat', {'all_pwr_train_on_real': all_pwr})
    print('done')

