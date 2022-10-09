load('all_acc_train_on_real3.mat');
load('all_pwr_train_on_real3.mat');
load('all_pwr_train_on_synth3.mat');
load('all_acc_train_on_synth3.mat');

num_data = 10:10:200;
% plot(num_data, all_acc_train_on_real3(1, i), '--s', "Color", "#0072BD");
% hold on
% plot(num_data, all_pwr_train_on_real3(1, i), '-o', "Color", "#0072BD");
% hold on
% plot(num_data, all_acc_train_on_synth3(1, i), '--s', "Color", "#A2142F");
% hold on
% plot(num_data, all_pwr_train_on_synth3(1, i), '-o', "Color", "#A2142F");
% grid on

plot(num_data, all_acc_train_on_real3(2, :), '--s', "Color", "#0072BD");
hold on
plot(num_data, all_pwr_train_on_real3(2, :), '-o', "Color", "#0072BD");
hold on
plot(num_data, all_acc_train_on_synth3(2, :), '--s', "Color", "#A2142F");
hold on
plot(num_data, all_pwr_train_on_synth3(2, :), '-o', "Color", "#A2142F");
grid on

xlabel('Number of Training Data Points');
ylabel('Accuracy / Relative Receive Power');

ylim([0.3, 1]);
legend('Acc. (trained on real)', 'Power (trained on real)', 'Acc. (trained on synth.)', 'Power (trained on synth.)');