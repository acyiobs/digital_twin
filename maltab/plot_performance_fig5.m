clear;
clc;
load('result/all_acc_train_on_real.mat'); % train_model_real.py
load('result/all_pwr_train_on_real.mat');
load('result/all_acc_train_on_synth_measured.mat'); % train_model_synth_measured.py
load('result/all_pwr_train_on_synth_measured.mat');
load('result/all_acc_train_on_synth_uniform.mat'); % train_model_synth_uniform.py
load('result/all_pwr_train_on_synth_uniform.mat');

all_acc_train_on_real = squeeze(mean(all_acc_train_on_real, 2));
all_pwr_train_on_real = squeeze(mean(all_pwr_train_on_real, 2));
all_acc_train_on_synth_measured = squeeze(mean(all_acc_train_on_synth_measured, 2));
all_pwr_train_on_synth_measured = squeeze(mean(all_pwr_train_on_synth_measured, 2));
all_acc_train_on_synth_uniform = squeeze(mean(all_acc_train_on_synth_uniform, 2));
all_pwr_train_on_synth_uniform = squeeze(mean(all_pwr_train_on_synth_uniform, 2));

num_data = 10:10:200;

figure;
plot(num_data, all_acc_train_on_real(2, :), '--s', "Color", "#0072BD");
hold on
plot(num_data, all_pwr_train_on_real(2, :), '-o', "Color", "#0072BD");
hold on
plot(num_data, all_acc_train_on_synth_measured(2, :), '--s', "Color", "#A2142F");
hold on
plot(num_data, all_pwr_train_on_synth_measured(2, :), '-o', "Color", "#A2142F");
hold on
plot(num_data, all_acc_train_on_synth_uniform(2, :), '--s', "Color", "#ff13a6");
hold on
plot(num_data, all_pwr_train_on_synth_uniform(2, :), '-o', "Color", "#ff13a6");
grid on
ylim([0.3, 1]);

xlabel('Number of Training Data Points');
ylabel('Accuracy / Relative Receive Power');
legend('Acc. (trained on real)', 'Power (trained on real)', 'Acc. (trained on synth.) measured codebook', 'Power (trained on synth.) measured codebook', 'Acc. (trained on synth.) uniform codebook', 'Power (trained on synth.) uniform codebook');

