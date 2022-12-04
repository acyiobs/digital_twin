load('all_acc_train_on_real_compare_to_transfer2.mat');
load('all_pwr_train_on_real_compare_to_transfer2.mat');
load('all_acc_train_on_transfer_final_batch8.mat');
load('all_pwr_train_on_transfer_final_batch8.mat');
load('all_acc_train_on_transfer_final_batch8_beamsteer2.mat');
load('all_pwr_train_on_transfer_final_batch8_beamsteer2.mat');

all_acc_train_on_real4_compare_to_transfer = squeeze(mean(all_acc_train_on_real_compare_to_transfer2, 2));
all_pwr_train_on_real4_compare_to_transfer = squeeze(mean(all_pwr_train_on_real_compare_to_transfer2, 2));
all_acc_train_on_transfer_final = squeeze(mean(all_acc_train_on_transfer_final_batch8, 2));
all_pwr_train_on_transfer_final = squeeze(mean(all_pwr_train_on_transfer_final_batch8, 2));
all_acc_train_on_transfer_final_beamsteer = squeeze(mean(all_acc_train_on_transfer_final_batch8_beamsteer2, 2));
all_pwr_train_on_transfer_final_beamsteer = squeeze(mean(all_pwr_train_on_transfer_final_batch8_beamsteer2, 2));


num_data = 0:5:100;

plot(5:5:100, all_acc_train_on_real4_compare_to_transfer(2, :), '--s', "Color", "#0072BD");
hold on
plot(5:5:100, all_pwr_train_on_real4_compare_to_transfer(2, :), '-o', "Color", "#0072BD");
hold on
plot(0:5:100, all_acc_train_on_transfer_final(2, :), '--s', "Color", "#000000");
hold on
plot(0:5:100, all_pwr_train_on_transfer_final(2, :), '-o', "Color", "#000000");
grid on
plot(0:5:100, all_acc_train_on_transfer_final_beamsteer(2, :), '--s', "Color", "#946801");
hold on
plot(0:5:100, all_pwr_train_on_transfer_final_beamsteer(2, :), '-o', "Color", "#946801");
grid on

xlabel('Number of Real Data Points Used for Training');
ylabel('Accuracy / Relative Receive Power');

ylim([0.25, 1]);
% legend( 'Acc. (transfer learining)', 'Power (transfer learining)');
legend('Acc. (trained on real)', 'Power (trained on real)', ...
    'Acc. (transfer learining) measured codebook', 'Power (transfer learining) measured codebook', ...
    'Acc. (transfer learining) unifrom codebook', 'Power (transfer learining) unifrom codebook');


axes('position', [0.25 0.25 0.5 0.5]);
box on
idx = (0:5:100) < 21;
plot(0:5:20, all_acc_train_on_transfer_final(2, 1:5), '--s', "Color", "#000000");
hold on
plot(0:5:20, all_pwr_train_on_transfer_final(2, 1:5), '-o', "Color", "#000000");
grid on
plot(0:5:20, all_acc_train_on_transfer_final_beamsteer(2, 1:5), '--s', "Color", "#946801");
hold on
plot(0:5:20, all_pwr_train_on_transfer_final_beamsteer(2, 1:5), '-o', "Color", "#946801");
grid on
axis tight

