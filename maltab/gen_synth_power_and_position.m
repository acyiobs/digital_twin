load('E:\Shuaifeng-Jiang\GitHub\digital_twin\DeepMIMOv2\deepsense_s1_synth_v3_interval_0p1.mat');
load('E:\Shuaifeng-Jiang\GitHub\digital_twin\real_beam_pwr.mat');
load('E:\Shuaifeng-Jiang\GitHub\digital_twin\ue_relative_pos.mat');
load('E:\Shuaifeng-Jiang\GitHub\digital_twin\codebook_beams\beam_angles.mat');

%% get best beam in the digital twin
beam_angle_z = 90*ones(64,1);
beam_angle_x = beam_anlges/pi*180;
beam_angle = [beam_angle_z, beam_angle_x.'];
F_CB = beam_steering_codebook2(beam_angle, 1, 16, 0);
F_CB(:, 1) = 0;
F_CB(:, end) = 0;

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




