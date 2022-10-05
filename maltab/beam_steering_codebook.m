function codebook = beam_steering_codebook(angles, num_z, num_x)
% angles:   [[z_angle1, x_angle1], [z_angle2, x_angle2], ...]
% num_x: number of horizontal antenna elements
% cross_pol: set to 1 if cross polarization is enabled

d = 0.5;
k_z = 0:num_z-1;
k_x = 0:num_x-1;

codebook = [];

for beam_idx=1:size(angles, 1)
    z_angle = angles(beam_idx, 1);
    x_angle = angles(beam_idx, 2);
    bf_vector_z = exp(1j*2*pi*k_z*d*cosd(z_angle));
    bf_vector_x = exp(1j*2*pi*k_x*d*cosd(x_angle));
    bf_vector = bf_vector_z.' * bf_vector_x;
    bf_vector = bf_vector(:);
    codebook = [codebook, bf_vector];
end

end

