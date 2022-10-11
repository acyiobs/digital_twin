load('all_acc_train_on_real_compare_to_transfer.mat');
load('all_pwr_train_on_real_compare_to_transfer.mat');
load('all_acc_train_on_transfer_final.mat');
load('all_pwr_train_on_transfer_final.mat');

all_acc_train_on_real4_compare_to_transfer = squeeze(mean(all_acc_train_on_real4_compare_to_transfer, 2));
all_pwr_train_on_real4_compare_to_transfer = squeeze(mean(all_pwr_train_on_real4_compare_to_transfer, 2));
all_acc_train_on_transfer_final = squeeze(mean(all_acc_train_on_transfer_final, 2));
all_pwr_train_on_transfer_final = squeeze(mean(all_pwr_train_on_transfer_final, 2));


num_data = 0:5:100;

plot(5:5:100, all_acc_train_on_real4_compare_to_transfer(2, :), '--s', "Color", "#0072BD");
hold on
plot(5:5:100, all_pwr_train_on_real4_compare_to_transfer(2, :), '-o', "Color", "#0072BD");
hold on
plot(0:5:100, all_acc_train_on_transfer_final(2, :), '--s', "Color", "#000000");
hold on
plot(0:5:100, all_pwr_train_on_transfer_final(2, :), '-o', "Color", "#000000");
grid on

xlabel('Number of Real Data Points Used for Training');
ylabel('Accuracy / Relative Receive Power');

ylim([0, 1]);
legend( 'Acc. (transfer learining)', 'Power (transfer learining)');
% legend('Acc. (trained on real)', 'Power (trained on real)', 'Acc. (transfer_final learining)', 'Power (transfer_final learining)');