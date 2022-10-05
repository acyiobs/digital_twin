clear;
clc;
load('data/deepsense_s1_synth_v3_interval_0p1.mat');
load('data/real_beam_pwr.mat');
load('data/ue_relative_pos.mat');
load('codebook_beams/beam_angles.mat');

%% get best beam in the digital twin
beam_angle_z = 90*ones(64,1);
beam_angle_x = beam_anlges/pi*180;
beam_angle = [beam_angle_z, beam_angle_x.'];
F_CB = beam_steering_codebook(beam_angle, 1, 16);
F_CB(:, 1) = 0;
F_CB(:, end) = 0;

F_CB = F_CB(:,2:4:64);
real_beam_pwr = real_beam_pwr(:,2:4:64);
num_beam = size(real_beam_pwr, 2);

dataset_synth = DeepMIMO_dataset{1,1}.user;
num_synth_UE = size(dataset_synth, 2);
num_rx = size(dataset_synth{1}.channel, 1);
num_tx = size(dataset_synth{1}.channel, 2);
num_subcarrier = size(dataset_synth{1}.channel, 3);

all_channel = zeros(num_synth_UE, num_tx, num_subcarrier); % num_rx is 1

synth_UE_loc = zeros(num_synth_UE, 2);

for i=1:num_synth_UE
    all_channel(i, :, :) = dataset_synth{i}.channel;
    synth_UE_loc(i,:) = dataset_synth{i}.loc(1:2);
end
all_channel = permute(all_channel, [1,3,2]); % -> num_synth_UE, num_subcarrier, num_tx

all_channel = reshape(all_channel, [], num_tx);
synth_beam_power = all_channel * F_CB; % -> (num_synth_UE, num_subcarrier), num_beam
synth_beam_power = reshape(synth_beam_power, num_synth_UE, num_subcarrier, []);
synth_beam_power = squeeze(sum(abs(synth_beam_power).^2, 2));

tmp = [];

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

synth_beam_power_nearest = synth_beam_power(nearest_synth_idx,:);

idx_diff = zeros(size(real_beam_pwr, 1), 1);

top1_acc = zeros(size(real_beam_pwr, 1), 1);
top2_acc = zeros(size(real_beam_pwr, 1), 1);
top3_acc = zeros(size(real_beam_pwr, 1), 1);
top4_acc = zeros(size(real_beam_pwr, 1), 1);
top5_acc = zeros(size(real_beam_pwr, 1), 1);

top1_pwr = zeros(size(real_beam_pwr, 1), 1);
top2_pwr = zeros(size(real_beam_pwr, 1), 1);
top3_pwr = zeros(size(real_beam_pwr, 1), 1);
top4_pwr = zeros(size(real_beam_pwr, 1), 1);
top5_pwr = zeros(size(real_beam_pwr, 1), 1);

for i=1:size(real_beam_pwr, 1)
    [real_best_pwr, real_best_idx] = max(real_beam_pwr(i, :));
    nearest_neighbor_pwr = synth_beam_power_nearest(i, :);
    nearest_neighbor_pwr = [nearest_neighbor_pwr; 1:num_beam].';
    nearest_neighbor_pwr = sortrows(nearest_neighbor_pwr, 1, 'descend');

    idx_diff(i) = nearest_neighbor_pwr(1, 2) - real_best_idx;

    top1_acc(i) = any(nearest_neighbor_pwr(1:1, 2) == real_best_idx);
    top2_acc(i) = any(nearest_neighbor_pwr(1:2, 2) == real_best_idx);
    top3_acc(i) = any(nearest_neighbor_pwr(1:3, 2) == real_best_idx);
    top4_acc(i) = any(nearest_neighbor_pwr(1:4, 2) == real_best_idx);
    top5_acc(i) = any(nearest_neighbor_pwr(1:5, 2) == real_best_idx);

    top1_pwr(i) = max(real_beam_pwr(i, nearest_neighbor_pwr(1:1, 2))) / real_best_pwr;
    top2_pwr(i) = max(real_beam_pwr(i, nearest_neighbor_pwr(1:2, 2))) / real_best_pwr;
    top3_pwr(i) = max(real_beam_pwr(i, nearest_neighbor_pwr(1:3, 2))) / real_best_pwr;
    top4_pwr(i) = max(real_beam_pwr(i, nearest_neighbor_pwr(1:4, 2))) / real_best_pwr;
    top5_pwr(i) = max(real_beam_pwr(i, nearest_neighbor_pwr(1:5, 2))) / real_best_pwr;
end

[~, real_best_idx] = max(real_beam_pwr, [], 2);
[~, synth_best_idx] = max(synth_beam_power_nearest, [], 2);

topk_acc_16beam = mean([top1_acc, top2_acc, top3_acc, top4_acc, top5_acc], 1);
topk_pwr_16beam = mean([top1_pwr, top2_pwr, top3_pwr, top4_pwr, top5_pwr], 1);

%% best beam vs position
figure(1);
plot(synth_best_idx)
hold on
plot(real_best_idx)
grid on
legend('nearest neighbor in digital replica','real');
xlabel("Time Sample Index (veichle moves from left to right)")
ylabel("Optimal Beam Index")
xlim([1, 116])

%% top-k performance
figure(2);
b1 = bar(categorical({'Accuracy' 'Relative Receive Power'}), [topk_acc_16beam; topk_pwr_16beam]);
ylabel('Performance');
xlim({'Accuracy' 'Relative Receive Power'});
ylim([0.4, 1]);
legend('Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5');
grid on;
