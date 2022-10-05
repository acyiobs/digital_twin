% load('deepsense_s1_synth_interval_0p1.mat');
load('real_beam_pwr.mat')
load('ue_relative_pos.mat')

%% get best beam in the digital twin
[F_CB, all_beams] = UPA_codebook_generator_abdul(16,1,1,4,1,1,0.5);
F_CB(:, 1) = 0;
F_CB(:, end) = 0;
F_CB = flip(F_CB, 2);

dataset_synth = DeepMIMO_dataset{1,1}.user;
num_synth_UE = size(dataset_synth, 2);
num_rx = size(dataset_synth{1}.channel, 1);
num_tx = size(dataset_synth{1}.channel, 2);
num_subcarrier = size(dataset_synth{1}.channel, 3);

all_channel = zeros(num_synth_UE, num_tx, num_subcarrier); % num_rx is 1

synth_UE_beam_pwr = zeros(num_synth_UE, 64, 64); % (ue, tx_beam, subcarrier)
synth_UE_loc = zeros(num_synth_UE, 2);

for i=1:num_synth_UE
    all_channel(i, :, :) = dataset_synth{i}.channel;
    synth_UE_loc(i,:) = dataset_synth{i}.loc(1:2);
end
all_channel = permute(all_channel, [1,3,2]); % -> num_synth_UE, num_subcarrier, num_tx

all_channel = reshape(all_channel, [], num_tx);
all_beam_power = all_channel * F_CB; % -> (num_synth_UE, num_subcarrier), num_beam
all_beam_power = reshape(all_beam_power, num_synth_UE, num_subcarrier, []);
all_beam_power = squeeze(sum(abs(all_beam_power).^2, 2));

[~, synth_UE_beam_idx] = max(all_beam_power , [], 2);

%% get best beam digital twin nearest neighbor in the real world
[~, real_UE_beam_idx] = max(real_beam_pwr, [], 2);


nearest_synth_idx = zeros(size(ue_relative_pos,1),1);
min_dist = zeros(size(ue_relative_pos,1),1);
for i=1:size(ue_relative_pos, 1)
    ue_pos = ue_relative_pos(i,:);
    dist_square = sum(abs(ue_pos - synth_UE_loc).^2, 2);
    [tmp1, tmp2] = min(dist_square);
    min_dist(i) = tmp1;
    nearest_synth_idx(i) = tmp2;  
end
min_dist = sqrt(min_dist);

synth_UE_beam_idx_reordered = synth_UE_beam_idx(nearest_synth_idx);

tmp = [synth_UE_beam_idx_reordered,  real_UE_beam_idx];

plot(synth_UE_beam_idx_reordered)
hold on
plot(real_UE_beam_idx)
legend('synthetic','real')
title('Synth vs. Real Datat: Optimal BS Beam Index')
xlabel("Position Index")
ylabel("Optimal Beam Index")
xlim([1,size(ue_relative_pos,1)])
ylim([1,size(F_CB,2)])
hold off

