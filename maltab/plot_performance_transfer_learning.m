load('all_acc_train_on_real4.mat');
load('all_pwr_train_on_real4.mat');
load('all_acc_train_on_transfer.mat');
load('all_pwr_train_on_transfer.mat');

all_acc_train_on_real4 = squeeze(mean(all_acc_train_on_real4, 2));
all_pwr_train_on_real4 = squeeze(mean(all_pwr_train_on_real4, 2));
all_acc_train_on_transfer = squeeze(mean(all_acc_train_on_transfer, 2));
all_pwr_train_on_transfer = squeeze(mean(all_pwr_train_on_transfer, 2));


num_data = 10:10:200;
% plot(num_data, all_acc_train_on_real3(1, i), '--s', "Color", "#0072BD");
% hold on
% plot(num_data, all_pwr_train_on_real3(1, i), '-o', "Color", "#0072BD");
% hold on
% plot(num_data, all_acc_train_on_synth3(1, i), '--s', "Color", "#A2142F");
% hold on
% plot(num_data, all_pwr_train_on_synth3(1, i), '-o', "Color", "#A2142F");
% grid on

plot(num_data, all_acc_train_on_real4(2, :), '--s', "Color", "#0072BD");
hold on
plot(num_data, all_pwr_train_on_real4(2, :), '-o', "Color", "#0072BD");
hold on
plot(num_data, all_acc_train_on_transfer(2, :), '--s', "Color", "#000000");
hold on
plot(num_data, all_pwr_train_on_transfer(2, :), '-o', "Color", "#000000");
grid on

xlabel('Number of Real Data Points Used for Training');
ylabel('Accuracy / Relative Receive Power');

ylim([0.4, 1]);
legend('Acc. (trained on real)', 'Power (trained on real)', 'Acc. (transfer learining)', 'Power (transfer learining)');