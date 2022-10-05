% load('deepsense_s1_synth_interval_0p1.mat');
load('real_beam_pwr.mat')
load('ue_relative_pos.mat')

%% get best beam in the digital twin
[F_CB, all_beams] = UPA_codebook_generator_abdul(16,1,1,4,1,1,0.5);
F_CB(:, 1) = 0;
F_CB(:, end) = 0;
F_CB = flip(F_CB, 2);

F_CB = F_CB(:,1:1:64);
real_beam_pwr = real_beam_pwr(:, 1:1:64);
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

[~, synth_UE_beam_idx] = max(synth_beam_power , [], 2);

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

topk_acc = [mean(top1_acc), mean(top2_acc), mean(top3_acc), mean(top5_acc)]
topk_pwr = [mean(top1_pwr), mean(top2_pwr), mean(top3_pwr), mean(top5_pwr)]
mean_idx_diff = mean(abs(idx_diff))

[~, real_best_idx] = max(real_beam_pwr, [], 2);
[~, synth_best_idx] = max(synth_beam_power_nearest, [], 2);

figure(1);
plot(synth_best_idx)
hold on
plot(real_best_idx)
legend('synthetic','real')
title('Synth vs. Real Datat: Optimal BS Beam Index')
xlabel("Position Index")
ylabel("Optimal Beam Index")
xlim([1,size(ue_relative_pos,1)])
xlim([1,200])
ylim([1,size(F_CB,2)])
hold off

figure(2);
histogram(abs(idx_diff), 0:20, 'Normalization', 'pdf')
grid on
xlabel('Beam Index Difference')
ylabel('Frequency')

beam16_acc = [0.3982    0.6711    0.7665    0.8669];
beam32_acc = [0.2111    0.4085    0.5570    0.7540];
beam64_acc = [0.1045    0.2053    0.3111    0.4844];

beam16_pwr = [0.6888    0.8609    0.9110    0.9528];
beam32_pwr = [0.6465    0.7624    0.8491    0.9448];
beam64_pwr = [0.6225    0.6851    0.7431    0.8352];

figure(3)
barcolors = validatecolor({'#9BE0D1','#D0E8CC','#F5F1DE','#E9C0A4','#DE8A8A'}, 'multiple');
x = categorical({'16 beams','32 beams','64 beams'});
bar(x, [beam16_acc; beam32_acc; beam64_acc]);
grid on;
legend('top-1', 'top-2', 'top-3', 'top-5');
ylabel('Accuray');

figure(4)
x = categorical({'16 beams','32 beams','64 beams'});
bar(x, [beam16_pwr; beam32_pwr; beam64_pwr]);
grid on;
legend('top-1', 'top-2', 'top-3', 'top-5');
ylabel('Relateive Receive Power');


%{
topk: top-1,2,3,5

16beam
topk_acc =
    0.3982    0.6711    0.7665    0.8669
topk_pwr =
    0.6888    0.8609    0.9110    0.9528
mean_idx_diff =
    0.7341

32beam
topk_acc =
    0.2111    0.4085    0.5570    0.7540
topk_pwr =
    0.6465    0.7624    0.8491    0.9448
mean_idx_diff =
    1.5068

64beam
topk_acc =
    0.1045    0.2053    0.3111    0.4844
topk_pwr =
    0.6225    0.6851    0.7431    0.8352
mean_idx_diff =
    3.1107
%}




