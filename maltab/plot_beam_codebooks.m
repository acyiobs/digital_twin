%% plot uniform codebook
load('codebook_beams\beam_angles.mat');

beam_angle_z = 90*ones(64,1);
beam_angle_x = beam_anlges/pi*180;

beam_angle_x = linspace(beam_angle_x(2),beam_angle_x(63),64);

beam_angle = [beam_angle_z, beam_angle_x.'];
F_CB = beam_steering_codebook2(beam_angle, 1, 16, 0);
F_CB = F_CB(:, 2:4:end);

num_tx = size(F_CB, 1);
d = 0.5; % antennna inteval in the lambda unit 
all_angles = (0:0.1:180)';

beam_power = zeros(size(F_CB, 2), size(all_angles, 1));
for beam_idx = 1:size(F_CB,2)
    beam = F_CB(:, beam_idx);
    for angle_idx = 1:size(all_angles,1)
        angle = all_angles(angle_idx);
        array_response = exp(-1j * 2 * pi * d * (0:num_tx-1) * cosd(angle));
        beam_power(beam_idx, angle_idx) = abs(array_response * beam).^2;
    end
end
beam_power = beam_power / max(beam_power(:));

figure(1);
% ax = figure(1);
% ax1 = subplot(1,2,1);
for i=1:size(beam_power, 1)
    polarplot(deg2rad(all_angles), beam_power(i,:), ':', 'linewidth', 1.2); % 10 * log10(beam_power(i, :)/max(beam_power(i, :)))
    grid on
    hold on
end
thetalim([0 180]);
title('Uniform Beam Codebook');

%% plot measured codebook
beam_set = 0:63;
num_of_beam = length(beam_set);

codebook_pattern = zeros(num_of_beam, 211);

for ii = 1:num_of_beam
    data = load(['codebook_beams\built_in_beam_pattern_',num2str(beam_set(num_of_beam-ii+1)),'.mat']);
    beam_pattern = data.beam_pattern;
    beam_pattern(1) = beam_pattern(2); % the first value is always instable
    codebook_pattern(ii, :) = beam_pattern;
end
codebook_pattern = codebook_pattern(2:4:num_of_beam, :);
codebook_pattern = codebook_pattern / max(codebook_pattern(:));

measurement_offset_angle=4*pi/180;
angle_start = pi-measurement_offset_angle;
angle_end = 0-measurement_offset_angle;
num_of_angle = length(beam_pattern);

figure(2);
for ii = 1:size(codebook_pattern,1)
    polarplot(linspace(angle_start, angle_end, num_of_angle), codebook_pattern(ii, :), 'linewidth', 1.2)
    grid on
    hold on
end
thetalim([0, 180]);
title('Measured Beam Codebook');