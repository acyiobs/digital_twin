load('deepsense_s1_synth_interval_0p1.mat');
load('real_beam_pwr.mat');
load('ue_relative_pos.mat');
load('E:\Shuaifeng-Jiang\GitHub\digital_twin\codebook_beams\beam_angles.mat');

%% get best beam in the digital twin
%[F_CB, all_beams] = UPA_codebook_generator_abdul(16,1,1,4,1,1,0.5);
beam_angle_z = 90*ones(64,1);
beam_angle_x = 180 - beam_anlges/pi*180;
beam_angle = [beam_angle_z, beam_angle_x.'];
F_CB = beam_steering_codebook2(beam_angle, 1, 16, 0);
F_CB(:, 1) = 0;
F_CB(:, end) = 0;
% F_CB = flip(F_CB, 2);

F_CB = F_CB(:,1:4:64);
real_beam_pwr = real_beam_pwr(:, 1:4:64);
num_beam = size(real_beam_pwr, 2);

dataset_synth = DeepMIMO_dataset{1,1}.user;
num_synth_UE = size(dataset_synth, 2);
num_rx = size(dataset_synth{1}.channel, 1);
num_tx = size(dataset_synth{1}.channel, 2);
num_subcarrier = size(dataset_synth{1}.channel, 3);

all_channel = zeros(num_synth_UE, num_tx, num_subcarrier); % num_rx is 1

synth_UE_beam_pwr = zeros(num_synth_UE, num_beam, 64); % (ue, tx_beam, subcarrier)
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

for UE_subsample=[1] % 1, 20, 50, 100, 200, 500
    %% sample the user grid
    first_grid_idx = 1:411*41;
    second_grid_idx = (411*41)+(1:51*470);

    first_grid_idx = reshape(first_grid_idx, 411, 41);
    second_grid_idx = reshape(second_grid_idx, 470, 51);

    first_grid_idx = first_grid_idx(1:UE_subsample:end, 1:UE_subsample:end);
    second_grid_idx = second_grid_idx(1:UE_subsample:end, 1:UE_subsample:end);
    first_grid_idx = reshape(first_grid_idx, 1, []);
    second_grid_idx = reshape(second_grid_idx, 1, []);
    all_idx = [first_grid_idx, second_grid_idx];

    synth_beam_power_ =  synth_beam_power(all_idx, :);
    synth_UE_loc_ = synth_UE_loc(all_idx, :);

    %% get best beam digital twin nearest neighbor in the real world
    [~, real_UE_beam_idx] = max(real_beam_pwr, [], 2);


    nearest_synth_idx = zeros(size(ue_relative_pos,1),1);
    min_dist = zeros(size(ue_relative_pos,1),1);
    for i=1:size(ue_relative_pos, 1)
        ue_pos = ue_relative_pos(i,:);
        dist_square = sum(abs(ue_pos - synth_UE_loc_).^2, 2);
        [tmp1, tmp2] = min(dist_square);
        min_dist(i) = tmp1;
        nearest_synth_idx(i) = tmp2;
    end
    min_dist = sqrt(min_dist);

    synth_beam_power_nearest = synth_beam_power_(nearest_synth_idx,:);

    idx_diff = zeros(size(real_beam_pwr, 1), 1);

    top1_acc = zeros(size(real_beam_pwr, 1), 1);
    top2_acc = zeros(size(real_beam_pwr, 1), 1);
    top3_acc = zeros(size(real_beam_pwr, 1), 1);
    top5_acc = zeros(size(real_beam_pwr, 1), 1);

    top1_pwr = zeros(size(real_beam_pwr, 1), 1);
    top2_pwr = zeros(size(real_beam_pwr, 1), 1);
    top3_pwr = zeros(size(real_beam_pwr, 1), 1);
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
        top5_acc(i) = any(nearest_neighbor_pwr(1:5, 2) == real_best_idx);

        top1_pwr(i) = max(real_beam_pwr(i, nearest_neighbor_pwr(1:1, 2))) / real_best_pwr;
        top2_pwr(i) = max(real_beam_pwr(i, nearest_neighbor_pwr(1:2, 2))) / real_best_pwr;
        top3_pwr(i) = max(real_beam_pwr(i, nearest_neighbor_pwr(1:3, 2))) / real_best_pwr;
        top5_pwr(i) = max(real_beam_pwr(i, nearest_neighbor_pwr(1:5, 2))) / real_best_pwr;
    end
    UE_subsample
    mean(top1_acc)
%     topk_acc = [mean(top1_acc), mean(top2_acc), mean(top3_acc), mean(top5_acc)]
%     topk_pwr = [mean(top1_pwr), mean(top2_pwr), mean(top3_pwr), mean(top5_pwr)]
%     mean_idx_diff = mean(abs(idx_diff))
[~, real_best_idx] = max(real_beam_pwr, [], 2);
[~, synth_best_idx] = max(synth_beam_power_nearest, [], 2);

figure(1);
plot(synth_best_idx)
hold on
plot(real_best_idx)
hold on
plot(real_best_idx-synth_best_idx);
legend('synthetic','real', 'diff');
title('Synth vs. Real Datat: Optimal BS Beam Index')
xlabel("Position Index")
ylabel("Optimal Beam Index")
xlim([0,200])

figure(2)
xlim([1,size(ue_relative_pos,1)])
xlim([1,200])
ylim([1,size(F_CB,2)])
hold off

end
