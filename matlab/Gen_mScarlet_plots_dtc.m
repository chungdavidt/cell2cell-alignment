% Gen_mScarlet_plots_dtc.m
%
% Per-slice counts-per-cell scatter, drawn in the stitched-canvas frame so it
% overlays stitch_slices.py's whole-slice channel TIFs pixel for pixel.
%
% Differs from Gen_mScarlet_plots.m in geometry only. The data path -- expmat
% column, QC mask, colour, clim, median-count printout -- is unchanged.
%
%   Gen_mScarlet_plots.m      this script
%   ---------------------     ---------------------------------------------
%   scatter(pos1, pos2)       scatter at pos*2 - (min_offset - 1)
%   axes aspect from figure   daspect [1 1 1]: one data unit is square
%   limits from the data      limits = the stitched canvas extent
%   marker 5 pt               marker CELL_DIAMETER_UM across, to scale
%   set(gca,'ydir','reverse') unchanged
%
% Frame. filt_neurons.pos is 40x-objective space at half the stitched pitch, so
% a cell's full-resolution canvas column is pos(:,1)*2 - (min_x_offset - 1),
% 1-indexed. FOV offsets come from the regression
% stitch_subslices.stitch_fov_channels uses (pos*2 = offset + pos40x), fitted
% over EVERY FOV with a row in the slice -- the rule stitch_slices.py applies,
% with no QC, marker or connected-component screen. The slope is left free at
% >= 3 rows and pinned at 1 below that, matching that script's min_rows = 1.
%
% The -1 in the canvas term reproduces the Python pipeline's off-by-one rather
% than correcting it: the images and export_subslice_cells.py's table both carry
% it, so keeping it is what makes these points land on them. It is one full-res
% pixel, 0.32 um. See reference_centroid_mapping_formula.md.
%
% Out of scope: the downsampled SUBSLICE frame. Its min offsets come from the
% marker-filtered FOV list in subslice_definitions.mat, which filt_neurons alone
% does not determine.
%
% Standalone by design. Gen_GCaMP_plots_dtc.m is the same script with
% GENE_COL, CLIM_MAX, OUT_DIR and BACKGROUND_CHANNEL retargeted -- fix one
% and fix the other. Their originals drifted apart exactly here.
%
% Requires filt_neurons in the workspace with fields expmat, slice, pos, pos40x
% and fov. Gen_mScarlet_plots.m needs only the first three.

%% ---------------------------------------------------------------- config

% expmat column, MATLAB 1-indexed. 114 mScarlet, 112 GCaMP. Index only -- the
% 114-gene panel labels these slots with stale gene names.
GENE_COL   = 114;
CLIM_MAX   = 10;              % 10 for mScarlet, 20 for GCaMP
OUT_DIR    = 'mScarlet_plots_dtc';

% Marker plotting is deliberately ungated on transcriptome quality; this is not
% the 20/5 cell-typing pair in local_config.py.
READS_THRESH = 0;
GENES_THRESH = 0;

% 'canvas_px' -- full-resolution stitched pixels, 1-indexed, overlays the TIF.
% 'um'        -- microns from the canvas origin.
FRAME = 'canvas_px';

% Divisor if the background image was resampled. 1 = full-resolution TIF.
% Ignored when FRAME is 'um'.
DOWNSAMPLE = 1;

EXVIVO_UM_PER_PX = 0.32;      % scope_profiles.EXVIVO_UM_PER_PX
FOV_SIZE         = 3200;      % preprocessing_config.FOV_SIZE
CELL_DIAMETER_UM = 10;        % marker diameter, drawn to scale

% Folder holding slice%d_MSCARLET.tif from stitch_slices.py. Empty draws no
% background and leaves the axes set up for one.
BACKGROUND_DIR = '';
BACKGROUND_CHANNEL = 'MSCARLET';

SLICES    = [];               % [] = every slice
WRITE_CSV = true;             % mapped coordinates + counts, per slice

%% ---------------------------------------------------------------- data

uniq_slices = unique(filt_neurons.slice);
uniq_slices = uniq_slices(~isnan(uniq_slices));
if ~isempty(SLICES)
    uniq_slices = uniq_slices(ismember(uniq_slices, SLICES));
end

countspercell = full(filt_neurons.expmat(:, GENE_COL));
total_cells   = numel(countspercell);
pass_qc       = sum(filt_neurons.expmat, 2) >= READS_THRESH & ...
                sum(filt_neurons.expmat > 0, 2) >= GENES_THRESH;
Percent_passed = nnz(pass_qc) / total_cells * 100 %#ok<NOPTS>
median(sum(filt_neurons.expmat(pass_qc, :), 2))

pos       = double(filt_neurons.pos);
pos40x    = double(filt_neurons.pos40x);
fov_names = string(filt_neurons.fov(:));

if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR);
end

counts = [];

for nn = 1:numel(uniq_slices)
    slice_id = uniq_slices(nn);
    in_slice = filt_neurons.slice == slice_id;

    % Canvas origin and extent, from every FOV in the slice.
    [min_x, min_y, canvas_w, canvas_h] = canvas_extent( ...
        pos, pos40x, fov_names, in_slice, FOV_SIZE);
    if isnan(min_x)
        fprintf('Slice %d: no FOV could be placed, skipped\n', slice_id);
        continue
    end

    % Full-resolution canvas position, 1-indexed. Same expression as
    % export_subslice_cells.py's x_raw / y_raw at DOWNSAMPLE_XY = 1.
    x_full = pos(:, 1) * 2 - (min_x - 1);
    y_full = pos(:, 2) * 2 - (min_y - 1);

    switch FRAME
        case 'canvas_px'
            x_fig = x_full / DOWNSAMPLE;
            y_fig = y_full / DOWNSAMPLE;
            x_lim = [0.5, canvas_w / DOWNSAMPLE + 0.5];
            y_lim = [0.5, canvas_h / DOWNSAMPLE + 0.5];
            unit_um   = EXVIVO_UM_PER_PX * DOWNSAMPLE;
            axis_label = 'stitched canvas (px)';
        case 'um'
            x_fig = (x_full - 0.5) * EXVIVO_UM_PER_PX;
            y_fig = (y_full - 0.5) * EXVIVO_UM_PER_PX;
            x_lim = [0, canvas_w * EXVIVO_UM_PER_PX];
            y_lim = [0, canvas_h * EXVIVO_UM_PER_PX];
            unit_um   = 1;
            axis_label = '\mum';
        otherwise
            error('FRAME must be ''canvas_px'' or ''um'', got ''%s''', FRAME);
    end

    keep     = in_slice & pass_qc;
    med_count = full(median(sum(filt_neurons.expmat(keep, :), 2)));
    disp('Slice number: ' + string(nn) + '   Median count: ' + string(med_count));

    %% ------------------------------------------------------------ figure

    fig = figure('Position', [50 50 600 400], 'Color', 'w');
    ax  = axes(fig); %#ok<LAXES>
    hold(ax, 'on');

    if ~isempty(BACKGROUND_DIR)
        bg_path = fullfile(BACKGROUND_DIR, ...
            sprintf('slice%d_%s.tif', slice_id, BACKGROUND_CHANNEL));
        if isfile(bg_path)
            % Truecolor, so the axes colormap stays free for the scatter.
            bg = imread(bg_path);
            if size(bg, 3) == 1
                [lo, hi] = robust_limits(bg);
                bg = min(max((double(bg) - lo) / (hi - lo), 0), 1);
                bg = repmat(bg, 1, 1, 3);
            else
                bg = double(bg) / double(intmax(class(bg)));
            end
            image(ax, 'XData', [x_lim(1) + 0.5 * diff(x_lim) / size(bg, 2), ...
                                x_lim(2) - 0.5 * diff(x_lim) / size(bg, 2)], ...
                      'YData', [y_lim(1) + 0.5 * diff(y_lim) / size(bg, 1), ...
                                y_lim(2) - 0.5 * diff(y_lim) / size(bg, 1)], ...
                      'CData', bg);
        else
            fprintf('  background not found: %s\n', bg_path);
        end
    end

    hs = scatter(ax, x_fig(keep), y_fig(keep), ...
        36, countspercell(keep), 'filled');

    clim(ax, [0 CLIM_MAX]);
    colorbar(ax);
    set(ax, 'YDir', 'reverse');
    % daspect, not `axis image`: that is equal + tight, and tight would throw
    % away the canvas extent in favour of the data's own bounding box.
    daspect(ax, [1 1 1]);
    set(ax, 'PlotBoxAspectRatioMode', 'auto');
    xlim(ax, x_lim);
    ylim(ax, y_lim);
    xlabel(ax, axis_label);
    ylabel(ax, axis_label);
    title(ax, sprintf('slice %u', slice_id));

    % Marker to scale. SizeData is points^2, so it has to be recomputed from
    % the axes whenever the figure resizes.
    d_data = CELL_DIAMETER_UM / unit_um;
    update_marker(ax, hs, d_data);
    set(fig, 'SizeChangedFcn', @(~, ~) update_marker(ax, hs, d_data));

    disp('_______________________________________________________________')
    counts = [counts; med_count]; %#ok<AGROW>

    savefig(fig, fullfile(OUT_DIR, sprintf('slice%d', slice_id)));
    exportgraphics(fig, fullfile(OUT_DIR, sprintf('slice%d.png', slice_id)), ...
        'Resolution', 300);

    if WRITE_CSV
        idx = find(keep);
        t = table(repmat(slice_id, numel(idx), 1), idx, ...
                  x_fig(idx), y_fig(idx), countspercell(idx), ...
            'VariableNames', {'slice_id', 'row_index_matlab', 'x', 'y', 'counts'});
        writetable(t, fullfile(OUT_DIR, sprintf('slice%d_cells.csv', slice_id)));
    end
end

%% ---------------------------------------------------------------- locals

function [min_x, min_y, canvas_w, canvas_h] = canvas_extent(pos, pos40x, ...
        fov_names, in_slice, fov_size)
% Canvas origin and size for one slice, matching stitch_fov_channels.
    fovs = unique(fov_names(in_slice));
    offs = nan(numel(fovs), 2);
    for k = 1:numel(fovs)
        rows = in_slice & fov_names == fovs(k);
        if ~any(rows)
            continue
        end
        offs(k, :) = fov_offset(pos(rows, :), pos40x(rows, :));
    end
    offs = offs(all(~isnan(offs), 2), :);
    if isempty(offs)
        [min_x, min_y, canvas_w, canvas_h] = deal(NaN);
        return
    end
    min_x    = min(offs(:, 1));
    min_y    = min(offs(:, 2));
    canvas_w = max(offs(:, 1)) + fov_size - min_x;
    canvas_h = max(offs(:, 2)) + fov_size - min_y;
end

function off = fov_offset(pos_fov, pos40x_fov)
% pos*2 = offset + pos40x. Slope free at >= 3 rows, pinned at 1 below that.
    n = size(pos_fov, 1);
    if n >= 3
        off = zeros(1, 2);
        for c = 1:2
            b = [ones(n, 1), pos40x_fov(:, c)] \ (pos_fov(:, c) * 2);
            off(c) = b(1);
        end
    elseif n >= 1
        off = mean(pos_fov * 2 - pos40x_fov, 1);
    else
        off = [NaN NaN];
        return
    end
    off = round_half_even(off);
end

function y = round_half_even(x)
% Python's round(): ties go to the even integer. MATLAB's round() goes away
% from zero, which would put an offset one pixel off the pipeline's on a tie.
    y = floor(x);
    frac = x - y;
    y = y + (frac > 0.5) + (frac == 0.5) .* (mod(y, 2) ~= 0);
end

function update_marker(ax, hs, d_data)
% SizeData for a marker d_data wide in data units, under an equal aspect ratio.
    if ~isvalid(ax) || ~isvalid(hs)
        return
    end
    u = get(ax, 'Units');
    set(ax, 'Units', 'points');
    p = get(ax, 'Position');
    set(ax, 'Units', u);
    scale = min(p(3) / diff(xlim(ax)), p(4) / diff(ylim(ax)));  % points per data unit
    set(hs, 'SizeData', max((d_data * scale) ^ 2, 1));
end

function [lo, hi] = robust_limits(img)
% 1st / 99.9th percentile off a strided subsample, no Statistics Toolbox.
    stride = max(1, round(numel(img) / 2e5));
    v = sort(double(img(1:stride:end)));
    lo = v(max(1, round(0.010 * numel(v))));
    hi = v(max(1, round(0.999 * numel(v))));
    if hi <= lo
        hi = lo + 1;
    end
end
