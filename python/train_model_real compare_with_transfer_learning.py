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
from data_feed import DataFeed


if __name__ == "__main__":
    real_beam_pwr_path = 'data/real_beam_pwr.mat'
    real_pos_path = 'data/ue_relative_pos.mat'
    synth_beam_pwr_path = 'data/synth_beam_power_measured.mat'
    synth_pos_path = 'data/synth_UE_loc.mat'

    torch.manual_seed(2022)
    now = datetime.datetime.now().strftime("%H_%M_%S")
    date = datetime.date.today().strftime("%y_%m_%d")
    comment = "synth" + now + "_" + date
    num_epoch = 80
    batch_size = 8
    val_batch_size =128

    all_acc = []
    all_pwr = []

    for num_data_point in np.arange(5, 101, 5): #[100, 150, 200, 300, 400, 500, 600, 700, 800]
        acc = []
        pwr = []
        for _ in range(30):
            rand_state = int(torch.randint(low=1, high=2000, size=(1,)))

            dataset_real_train = DataFeed(pos_path=real_pos_path, beam_pwr_path=real_beam_pwr_path, rand_state=rand_state, mode='train', num_data_point=num_data_point)
            dataset_real_test = DataFeed(pos_path=real_pos_path, beam_pwr_path=real_beam_pwr_path, rand_state=rand_state, mode='test')
            dataset_synth_train = DataFeed(pos_path=synth_pos_path, beam_pwr_path=synth_beam_pwr_path, rand_state=rand_state, mode='train', num_data_point=num_data_point)
            dataset_synth_test = DataFeed(pos_path=synth_pos_path, beam_pwr_path=synth_beam_pwr_path, rand_state=rand_state, mode='test')

            real_train_loader = DataLoader(dataset_real_train, batch_size=batch_size, shuffle=True)
            real_test_loader = DataLoader(dataset_real_test, val_batch_size, shuffle=False)
            synth_train_loader = DataLoader(dataset_synth_train, batch_size=batch_size, shuffle=True)
            synth_test_loader = DataLoader(dataset_synth_test, val_batch_size, shuffle=False)

            print('Number of data points : '+ str(num_data_point))
            test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, PATH = train_model(
                train_loader=real_train_loader,
                val_loader=real_test_loader,
                test_loader=real_test_loader,
                comment=comment,
                num_classes=16,
                num_epoch=num_epoch,
                if_writer=False,
            )
            acc.append(test_acc)
            pwr.append(test_pwr)
        acc = np.stack(acc, -1)
        pwr = np.stack(pwr, -1)

        all_acc.append(acc)
        all_pwr.append(pwr)

    all_acc = np.stack(all_acc, -1)
    all_pwr = np.stack(all_pwr, -1)

    print(all_acc)
    print(all_pwr)
    savemat('result/all_acc_train_on_real_compare_to_transfer.mat', {'all_acc_train_on_real_compare_to_transfer': all_acc})
    savemat('result/all_pwr_train_on_real_compare_to_transfer.mat', {'all_pwr_train_on_real_compare_to_transfer': all_pwr})
    print('done')

