load('all_acc_train_on_real.mat')
load('all_pwr_train_on_real.mat')
load('all_pwr_train_on_synth.mat')
load('all_acc_train_on_synth.mat')

per = 1928 * [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
plot(per, all_acc_train_on_real(1, :), '--s', "Color", "#0072BD");
hold on
plot(per, all_pwr_train_on_real(1, :), '-o', "Color", "#0072BD");
hold on
plot(per, all_acc_train_on_synth(1, :), '--s', "Color", "#A2142F");
hold on
plot(per, all_pwr_train_on_synth(1, :), '-o', "Color", "#A2142F");
grid on

% plot(per, all_acc_train_on_real(3, :), '--s', "Color", "#0072BD");
% hold on
% plot(per, all_pwr_train_on_real(3, :), '-o', "Color", "#0072BD");
% hold on
% plot(per, all_acc_train_on_synth(3, :), '--s', "Color", "#A2142F");
% hold on
% plot(per, all_pwr_train_on_synth(3, :), '-o', "Color", "#A2142F");
% grid on

xlabel('Number of Training Data Points');
ylabel('Accuracy / Relative Receive Power');

ylim([0.5, 1])
legend('Acc. (trained on real)', 'Power (trained on real)', 'Acc. (trained on synth.)', 'Power (trained on synth.)');