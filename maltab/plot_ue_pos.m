load('E:\Shuaifeng-Jiang\GitHub\digital_twin\DeepMIMOv2\deepsense_s1_synth_v3_interval_0p1.mat');
load('E:\Shuaifeng-Jiang\GitHub\digital_twin\real_beam_pwr.mat');
load('E:\Shuaifeng-Jiang\GitHub\digital_twin\ue_relative_pos.mat');

figure(1);

%% plot synth UE positions
synth_UE_loc = zeros(num_synth_UE, 2);
for i=1:num_synth_UE
    synth_UE_loc(i,:) = dataset_synth{i}.loc(1:2);
end
scatter(synth_UE_loc(:,1), synth_UE_loc(:,2));
hold on;
xlabel('X-coordinates (meter)');
ylabel('Y-coordinates (meter)');

%% plot real UE positions
scatter(ue_relative_pos(:,1), ue_relative_pos(:,2));
hold on;
%% plot BS positions
scatter(0, 0, 120,  'X');
grid on;
daspect([1 1 1]);
xlim([0 50]);
xlabel('X-coordinates (meter)');
ylabel('Y-coordinates (meter)');
legend('UE grid', 'UE', 'BS');