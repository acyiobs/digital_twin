clc;
clear;
load('result/all_acc_train_on_real_compare_to_transfer.mat'); %train_model_real compare_with_transfer_learning.py
load('result/all_pwr_train_on_real_compare_to_transfer.mat');
load('result/all_acc_train_on_transfer_measured.mat'); %train_model_transfer_learning_measured.py
load('result/all_pwr_train_on_transfer_measured.mat');
load('result/all_acc_train_on_transfer_uniform.mat'); %train_model_transfer_learning_uniform
load('result/all_pwr_train_on_transfer_uniform.mat');

all_acc_train_on_real_compare_to_transfer = squeeze(mean(all_acc_train_on_real_compare_to_transfer, 2));
all_pwr_train_on_real_compare_to_transfer = squeeze(mean(all_pwr_train_on_real_compare_to_transfer, 2));
all_acc_train_on_transfer_measured = squeeze(mean(all_acc_train_on_transfer_measured, 2));
all_pwr_train_on_transfer_measured = squeeze(mean(all_pwr_train_on_transfer_measured, 2));
all_acc_train_on_transfer_uniform = squeeze(mean(all_acc_train_on_transfer_uniform, 2));
all_pwr_train_on_transfer_uniform = squeeze(mean(all_pwr_train_on_transfer_uniform, 2));

num_data = 0:5:100;

plot(5:5:100, all_acc_train_on_real_compare_to_transfer(2, :), '--s', "Color", "#0072BD");
hold on
plot(5:5:100, all_pwr_train_on_real_compare_to_transfer(2, :), '-o', "Color", "#0072BD");
hold on
plot(0:5:100, all_acc_train_on_transfer_measured(2, :), '--s', "Color", "#7e2f8e");
hold on
plot(0:5:100, all_pwr_train_on_transfer_measured(2, :), '-o', "Color", "#7e2f8e");
grid on
plot(0:5:100, all_acc_train_on_transfer_uniform(2, :), '--s', "Color", "#946801");
hold on
plot(0:5:100, all_pwr_train_on_transfer_uniform(2, :), '-o', "Color", "#946801");
grid on

xlabel('Number of Real Data Points Used for Training');
ylabel('Accuracy / Relative Receive Power');

ylim([0.25, 1]);
legend('Acc. (trained on real)', 'Power (trained on real)', ...
    'Acc. (transfer learining) measured codebook', 'Power (transfer learining) measured codebook', ...
    'Acc. (transfer learining) uniform codebook', 'Power (transfer learining) uniform codebook');

% axes('Position',[.2 .2 .2 .2])
% box on
% plot(0:5:20, all_acc_train_on_transfer_measured(2, 1:5), '--s', "Color", "#7e2f8e");
% hold on
% plot(0:5:20, all_pwr_train_on_transfer_measured(2, 1:5), '-o', "Color", "#7e2f8e");
% grid on
% plot(0:5:20, all_acc_train_on_transfer_uniform(2, 1:5), '--s', "Color", "#946801");
% hold on
% plot(0:5:20, all_pwr_train_on_transfer_uniform(2, 1:5), '-o', "Color", "#946801");
% grid on


