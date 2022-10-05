% load('DeepSenseScenario1_synth.mat');
% load('real_beam_pwr.mat')
% load('ue_relative_pos.mat')
[F_CB,all_beams] = UPA_codebook_generator_abdul(16,1,1,4,1,1,0.5);
F_CB(:, 1) = 0;
F_CB(:, end) = 0;
F_CB = flip(F_CB, 2);

dataset_synth = DeepMIMO_dataset{1,1}.user;
num_synth_UE = size(dataset_synth, 2);

synth_UE_beam_pwr = zeros(num_synth_UE, 64, 64); % (ue, tx_beam, subcarrier)
synth_UE_beam_idx = zeros(num_synth_UE, 1);
synth_UE_loc = zeros(num_synth_UE, 2);

for i=1:num_synth_UE
    for c=1:64
        channel = squeeze(dataset_synth{i}.channel(:,:,c));
        beam_power = abs(channel * F_CB);
        beam_power = sum(beam_power, 1);
        synth_UE_beam_pwr(i,:,c) = beam_power;
    end
    beam_power_sum = squeeze(sum(synth_UE_beam_pwr(i,:,:), 3));
    [M, I] = max(beam_power_sum,[],"all","linear");
    %[ue_beam_idx, bs_beam_idx] = ind2sub(size(beam_power_sum), I);
    synth_UE_beam_idx(i,:) = I;
    synth_UE_loc(i,:) = dataset_synth{i}.loc(1:2);
end

[~, real_UE_beam_idx] = max(real_beam_pwr, [], 2) ;
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

