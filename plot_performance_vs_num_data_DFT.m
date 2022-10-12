load('all_acc_train_on_real4.mat');
load('all_pwr_train_on_real4.mat');
load('all_acc_train_on_synth_DFT.mat');
load('all_pwr_train_on_synth_DFT.mat');


all_acc_train_on_real4 = squeeze(mean(all_acc_train_on_real4, 2));
all_pwr_train_on_real4 = squeeze(mean(all_pwr_train_on_real4, 2));
all_acc_train_on_synth4 = squeeze(mean(all_acc_train_on_synth_DFT, 2));
all_pwr_train_on_synth4 = squeeze(mean(all_pwr_train_on_synth_DFT, 2));

num_data = 10:10:200;
% plot(num_data, all_acc_train_on_real3(1, i), '--s', "Color", "#0072BD");
% hold on
% plot(num_data, all_pwr_train_on_real3(1, i), '-o', "Color", "#0072BD");
% hold on
% plot(num_data, all_acc_train_on_synth3(1, i), '--s', "Color", "#A2142F");
% hold on
% plot(num_data, all_pwr_train_on_synth3(1, i), '-o', "Color", "#A2142F");
% grid on
figure;
plot(num_data, all_acc_train_on_real4(2, :), '--s', "Color", "#0072BD");
hold on
plot(num_data, all_pwr_train_on_real4(2, :), '-o', "Color", "#0072BD");
hold on
plot(num_data, all_acc_train_on_synth4(2, :), '--s', "Color", "#A2142F");
hold on
plot(num_data, all_pwr_train_on_synth4(2, :), '-o', "Color", "#A2142F");
grid on

xlabel('Number of Training Data Points');
ylabel('Accuracy / Relative Receive Power');

ylim([0.3, 1]);
legend('Acc. (trained on real)', 'Power (trained on real)', 'Acc. (trained on synth.)', 'Power (trained on synth.)');