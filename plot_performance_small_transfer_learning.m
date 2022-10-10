% load('all_acc_train_on_real4.mat');
% load('all_pwr_train_on_real4.mat');
load('all_acc_train_on_transfer_small.mat');
load('all_pwr_train_on_transfer_small.mat');

all_acc_train_on_real4 = squeeze(mean(all_acc_train_on_real4, 2));
all_pwr_train_on_real4 = squeeze(mean(all_pwr_train_on_real4, 2));
all_acc_train_on_transfer_small = squeeze(mean(all_acc_train_on_transfer_small, 2));
all_pwr_train_on_transfer_small = squeeze(mean(all_pwr_train_on_transfer_small, 2));


num_data = 2:2:18;

% plot(num_data, all_acc_train_on_real4(2, :), '--s', "Color", "#0072BD");
% hold on
% plot(num_data, all_pwr_train_on_real4(2, :), '-o', "Color", "#0072BD");
% hold on
plot(num_data, all_acc_train_on_transfer_small(2, :), '--s', "Color", "#000000");
hold on
plot(num_data, all_pwr_train_on_transfer_small(2, :), '-o', "Color", "#000000");
grid on

xlabel('Number of Real Data Points Used for Training');
ylabel('Accuracy / Relative Receive Power');

ylim([0.4, 1]);
legend( 'Acc. (transfer learining)', 'Power (transfer learining)');
% legend('Acc. (trained on real)', 'Power (trained on real)', 'Acc. (transfer_small learining)', 'Power (transfer_small learining)');