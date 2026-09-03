clear all
close all

load('./stitched_hyb.mat');
align = readtable('./alignment.csv');

slide_id = 2;
slices = 8;
subset = align(align.Slide == 2, :);
dims = size(JH302_2_9);
volume = zeros(dims(1), dims(2), slices);

for j = 1:slices
    idx = align.Slide == slide_id & align.Slice == j;
    x_px = double(align{idx, "X_px_"});
    y_px = double(align{idx, "Y_px_"});
    angle = double(align{idx,"Degrees"});
    table_name = "JH302_" + string(slide_id) + "_" + string(j);
    rotated = rotate_shift_matrix(evalin("base", table_name), angle, x_px, y_px);
    volume(:,:,j) = rotated;
    clear rotated
end

scale_x = 1/6;
scale_y = 1/6;
scale_z = 1;  % no downsampling in z

% Downsample using cubic interpolation (good balance of sharpness vs aliasing)
volume_downsampled = imresize3(volume, [size(volume,1)*scale_x, size(volume,2)*scale_y, size(volume,3)*scale_z], 'cubic');