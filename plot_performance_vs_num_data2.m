load('all_acc_train_on_real2.mat')
load('all_pwr_train_on_real2.mat')
load('all_pwr_train_on_synth2.mat')
load('all_acc_train_on_synth2.mat')

num_data = [100, 150, 200, 300, 400, 500, 600, 700, 800];
i = [2,3,4,5,6,7,8,9,10];
% plot(num_data, all_acc_train_on_real2(1, i), '--s', "Color", "#0072BD");
% hold on
% plot(num_data, all_pwr_train_on_real2(1, i), '-o', "Color", "#0072BD");
% hold on
% plot(num_data, all_acc_train_on_synth2(1, i), '--s', "Color", "#A2142F");
% hold on
% plot(num_data, all_pwr_train_on_synth2(1, i), '-o', "Color", "#A2142F");
% grid on

plot(num_data, all_acc_train_on_real2(2, i), '--s', "Color", "#0072BD");
hold on
plot(num_data, all_pwr_train_on_real2(2, i), '-o', "Color", "#0072BD");
hold on
plot(num_data, all_acc_train_on_synth2(2, i), '--s', "Color", "#A2142F");
hold on
plot(num_data, all_pwr_train_on_synth2(2, i), '-o', "Color", "#A2142F");
grid on

xlabel('Number of Training Data Points');
ylabel('Accuracy / Relative Receive Power');

ylim([0.8, 1]);
legend('Acc. (trained on real)', 'Power (trained on real)', 'Acc. (trained on synth.)', 'Power (trained on synth.)');