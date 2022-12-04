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

% train_on_synth_measured_codebook = all_acc_train_on_real4_compare_to_transfer(2)

top_k = 1;
% num_data = 0:5:100;
% 
% plot(5:5:100, all_acc_train_on_real4_compare_to_transfer(top_k, :), '--s', "Color", "#0072BD");
% hold on
% plot(5:5:100, all_pwr_train_on_real4_compare_to_transfer(top_k, :), '-o', "Color", "#0072BD");
% hold on
% plot(0:5:100, all_acc_train_on_transfer_final(top_k, :), '--s', "Color", "#000000");
% hold on
% plot(0:5:100, all_pwr_train_on_transfer_final(top_k, :), '-o', "Color", "#000000");
% hold on
% plot(0:5:100, all_acc_train_on_transfer_final_beamsteer(top_k, :), '--s', "Color", "#946801");
% hold on
% plot(0:5:100, all_pwr_train_on_transfer_final_beamsteer(top_k, :), '-o', "Color", "#946801");
% hold on
% 
% xlabel('Number of Real Data Points Used for Training');
% ylabel('Accuracy / Relative Receive Power');
% 
% ylim([0.25, 1]);
% legend( 'Acc. (transfer learining)', 'Power (transfer learining)');
% legend('Acc. (trained on real)', ...
%     'Acc. (transfer learining) measured codebook', ...
%     'Acc. (transfer learining) unifrom codebook');

% acc
train_on_synth_m_acc = all_acc_train_on_transfer_final(top_k, 1);
train_on_synth_u_acc = all_acc_train_on_transfer_final_beamsteer(top_k, 1);
transfer_m_acc = all_acc_train_on_transfer_final(top_k, 5);
transfer_u_acc = all_acc_train_on_transfer_final_beamsteer(top_k, 5);
train_on_real_20_acc = all_acc_train_on_real4_compare_to_transfer(top_k, 4);
train_on_real_100_acc = all_acc_train_on_real4_compare_to_transfer(top_k, end);

% pwr
train_on_synth_m_pwr = all_pwr_train_on_transfer_final(top_k, 1);
train_on_synth_u_pwr = all_pwr_train_on_transfer_final_beamsteer(top_k, 1);
transfer_m_pwr = all_pwr_train_on_transfer_final(top_k, 5);
transfer_u_pwr = all_pwr_train_on_transfer_final_beamsteer(top_k, 5);
train_on_real_20_pwr = all_pwr_train_on_real4_compare_to_transfer(top_k, 4);
train_on_real_100_pwr = all_pwr_train_on_real4_compare_to_transfer(top_k, end);

% bars = [train_on_real_100_pwr, train_on_real_100_acc;
% train_on_real_20_pwr, train_on_real_20_acc;
% transfer_u_pwr, transfer_u_acc;
% transfer_m_pwr, transfer_m_acc;
% train_on_synth_u_pwr, train_on_synth_u_acc;
% train_on_synth_m_pwr, train_on_synth_m_acc;];
% 
% 
% barh(bars)
% xlim([0 1]);
% xlabel('Top-2 Accuracy Tested on Real-World Data');
% grid on;

acc_measured_cb = [train_on_synth_m_acc, transfer_m_acc-train_on_synth_m_acc];
acc_uniform_cb = [train_on_synth_u_acc, transfer_u_acc-train_on_synth_u_acc];
acc_train_on_real_20 = [train_on_real_20_acc, 0];
acc_train_on_real_100 = [train_on_real_100_acc, 0];

pwr_measured_cb = [train_on_synth_m_pwr, transfer_m_pwr-train_on_synth_m_pwr];
pwr_uniform_cb = [train_on_synth_u_pwr, transfer_u_pwr-train_on_synth_u_pwr];
pwr_train_on_real_20 = [train_on_real_20_pwr, 0];
pwr_train_on_real_100 = [train_on_real_100_pwr, 0];

% 
% % Y = [acc_train_on_real_100; acc_train_on_real_20; acc_measured_cb; acc_uniform_cb;];
% Y = [acc_uniform_cb;acc_measured_cb;acc_train_on_real_20;acc_train_on_real_100];
% 
% bar(Y, 'stacked');
% 
% grid on;

stackData = zeros(2,4,2);
stackData(1,:,:) = [pwr_train_on_real_100;pwr_train_on_real_20;pwr_measured_cb;pwr_uniform_cb;];
stackData(2,:,:) = [acc_train_on_real_100;acc_train_on_real_20;acc_measured_cb;acc_uniform_cb;];

groupLabels = {'Power', 'Accuracy'};

NumGroupsPerAxis = size(stackData, 1);
NumStacksPerGroup = size(stackData, 2);
% Count off the number of bins
groupBins = 1:NumGroupsPerAxis;
MaxGroupWidth = 0.8; % Fraction of 1. If 1, then we have all bars in groups touching
groupOffset = MaxGroupWidth/NumStacksPerGroup;
figure
    hold on; 
for i=1:NumStacksPerGroup
    Y = squeeze(stackData(:,i,:));
    
    % Center the bars:
    
    internalPosCount = i - ((NumStacksPerGroup+1) / 2);
    
    % Offset the group draw positions:
    groupDrawPos = (internalPosCount)* groupOffset + groupBins;
    
    h(i,:) = barh(Y, 'stacked');
    set(h(i,:),'BarWidth',0.8*groupOffset);
    set(h(i,:),'XData',groupDrawPos);
end
hold off;
set(gca,'YTickMode','manual');
set(gca,'YTick',1:NumGroupsPerAxis);
set(gca,'YTickLabelMode','manual');
set(gca,'YTickLabel',groupLabels);
grid on;
xlabel('Top-2 Performance Tested on Real-World Data')

