% load('Topk_acc_16beam.mat');
% load('Topk_pwr_16beam.mat');
% load('Topk_acc_64beam.mat');
% load('Topk_pwr_64beam.mat');
% % barcolors = ["#FFFFCC","#A1DAB4","#41B6C4","#2C7FB8","#253494"];
% barcolors = ["#9BE0D1","#D0E8CC","#F5F1DE","#E9C0A4","#DE8A8A"];
% 
% figure(2);
% ax1 = subplot(2,1,1);
% b1 = bar(categorical({'16 beam' '64 beam'}), [topk_acc_16beam;topk_acc_64beam]);
% ylabel('Accuracy');
% xlim({'16 beam' '64 beam'});
% grid on;
% 
% ax2 = subplot(2,1,2);
% b2 = bar(categorical({'16 beam' '64 beam'}), [topk_pwr_16beam;topk_pwr_64beam]);
% ylabel('Relative Receive Power');
% legend('Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5');
% xlim({'16 beam' '64 beam'});
% grid on;
% 
% for k = 1:5
%     b1(k).FaceColor = barcolors(k);
%     b2(k).FaceColor = barcolors(k);
% end

load('Topk_acc_16beam.mat');
load('Topk_pwr_16beam.mat');
barcolors = ["#9BE0D1","#D0E8CC","#F5F1DE","#E9C0A4","#DE8A8A"];

figure(2);
b1 = bar(categorical({'Accuracy' 'Relative Receive Power'}), [topk_acc_16beam;topk_pwr_16beam]);
ylabel('Performance');
xlim({'Accuracy' 'Relative Receive Power'});
ylim([0.4, 1]);
legend('Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5');
grid on;


for k = 1:5
    b1(k).FaceColor = barcolors(k);
    b2(k).FaceColor = barcolors(k);
end
