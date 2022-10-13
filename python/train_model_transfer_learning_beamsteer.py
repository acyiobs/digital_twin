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
    real_beam_pwr_path = 'real_beam_pwr.mat'
    real_pos_path = 'ue_relative_pos.mat'
    synth_beam_pwr_path = 'synth_beam_power_beamsteer2.mat'
    synth_pos_path = 'synth_UE_loc.mat'

    torch.manual_seed(2022)
    
    batch_size = 32
    val_batch_size =128

    all_acc = []
    all_pwr = []

    max_data_point = 1928
    for _ in range(30):
        now = datetime.datetime.now().strftime("%H_%M_%S")
        date = datetime.date.today().strftime("%y_%m_%d")
        comment = "transfer" + now + "_" + date
        rand_state = int(torch.randint(low=1, high=2000, size=(1,)))
        acc = []
        pwr = []
         
        
        dataset_real_test = DataFeed(pos_path=real_pos_path, beam_pwr_path=real_beam_pwr_path, rand_state=rand_state, mode='test')
        dataset_synth_train = DataFeed(pos_path=synth_pos_path, beam_pwr_path=synth_beam_pwr_path, rand_state=rand_state, mode='train', num_data_point=200)

        real_test_loader = DataLoader(dataset_real_test, val_batch_size, shuffle=False)
        synth_train_loader = DataLoader(dataset_synth_train, batch_size=batch_size, shuffle=True)

        test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, PATH = train_model(
            train_loader=synth_train_loader,
            val_loader=real_test_loader,
            test_loader=real_test_loader,
            comment=comment,
            num_classes=16,
            num_epoch=80,
            if_writer=True,
            )
        acc.append(test_acc)
        pwr.append(test_pwr)
        
        model_path = PATH

        for num_data_point in range(5, 101, 5):
            dataset_real_train = DataFeed(pos_path=real_pos_path, beam_pwr_path=real_beam_pwr_path, rand_state=rand_state, mode='train', num_data_point=num_data_point)
            real_train_loader = DataLoader(dataset_real_train, batch_size=8, shuffle=True)

            print('Number of data points : '+ str(num_data_point))
            test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, PATH = train_model(
                train_loader=real_train_loader,
                val_loader=real_test_loader,
                test_loader=real_test_loader,
                comment=comment,
                num_classes=16,
                num_epoch=40,
                if_writer=False,
                model_path=model_path,
                lr=1e-4,
            )
            acc.append(test_acc)
            pwr.append(test_pwr)
        acc = np.stack(acc, -1)
        pwr = np.stack(pwr, -1)

        all_acc.append(acc)
        all_pwr.append(pwr)

    all_acc = np.stack(all_acc, -1).swapaxes(-1, -2)
    all_pwr = np.stack(all_pwr, -1).swapaxes(-1, -2)

    print(all_acc)
    print(all_pwr)
    savemat('all_acc_train_on_transfer_final_batch8_beamsteer2.mat', {'all_acc_train_on_transfer_final_batch8_beamsteer2': all_acc})
    savemat('all_pwr_train_on_transfer_final_batch8_beamsteer2.mat', {'all_pwr_train_on_transfer_final_batch8_beamsteer2': all_pwr})
    print('done')


