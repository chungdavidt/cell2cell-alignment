function output = rotate_shift_matrix(input_matrix, theta_deg, shift_x, shift_y)
    % Convert angle to radians
    theta = deg2rad(theta_deg);

    % Size of the input matrix
    [rows, cols] = size(input_matrix);

    % Center of the image
    cx = (cols + 1) / 2;
    cy = (rows + 1) / 2;

    % Output matrix (same size)
    output = zeros(rows, cols);

    % Inverse rotation matrix
    R = [cos(theta), sin(theta); -sin(theta), cos(theta)];

    for i = 1:rows
        for j = 1:cols
            % Shifted destination coordinates relative to center
            x = j - cx - shift_x;
            y = i - cy - shift_y;

            % Map back to source coordinates using inverse rotation
            src = R * [x; y];
            x_src = src(1) + cx;
            y_src = src(2) + cy;

            % Bilinear interpolation
            if x_src >= 1 && x_src <= cols && y_src >= 1 && y_src <= rows
                output(i, j) = bilinear_interp(input_matrix, x_src, y_src);
            else
                output(i, j) = 0;
            end
        end
    end
end
