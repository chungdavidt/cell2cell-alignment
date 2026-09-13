function result = plot_marker_slices(filt_neurons, cfg)
% plot_marker_slices -- per-slice marker scatter for one marker and one set of
% thresholds.
%
%   result = plot_marker_slices(filt_neurons, cfg)
%
% Called by gen_marker_plots_dtc.m (one run) and sweep_marker_plots_dtc.m
% (every combination). cfg holds exactly the settings gen_marker_plots_dtc.m
% defines and documents; a missing or unknown field is an error.
%
% Colour is stepwise and absolute. The count -> colour mapping is built first,
% over a span fixed per marker at [0, RAMP_MAX], one discrete level per integer
% count, so a zero-count cell sits at the bottom colour as in Gen_*_plots.m;
% MIN_ROLONIES then chooses which cells are displayed, and the colorbar is
% cropped to start at it. A 9-rolony cell draws the same colour at a cutoff of
% 0 and at a cutoff of 5, and the ramp's slope never changes. COLORMAP picks
% the palette: any MATLAB colormap name (parula is the blue -> yellow default),
% or 'marker' for the anchor colours in the project's marker_profiles.py.
%
% Frame. A slice's window is the same in every figure of it, whatever the QC
% pair, cutoff, marker or crop. Its centre is the midpoint of the bounding box
% of every cell in that slice, before any filter. Its width and height are one
% span for all slices: the widest slice measured the same way, or AXIS_SPAN_UM
% when set. Axes are in µm, pos * 2 * EXVIVO_UM_PER_PX -- one pos unit is two
% 40x pixels -- with EXVIVO_UM_PER_PX read from scope_profiles.py. The origin
% is the corner of that section's stitched image, not a tissue landmark.
%
% Writes to
%   <ANALYSIS_ROOT>\preprocessing\<Marker>_plots_dtc\qc<reads>_<genes>\<crop|full>\ge<cut>_sat<cap>\
% one .fig and one .png per slice, median_total_reads.csv, and last
% plot_settings.txt. One level per filter, in the order they apply: QC and the
% crop set which cells exist, the cutoff which of them are drawn. COLORMAP,
% DRAW_BELOW_CUTOFF, SUBSLICE_DEFINITIONS_OVERRIDE, AXIS_SPAN_UM and the figure
% settings are not in the path; plot_settings.txt records them.
%
% SKIP_EXISTING skips a folder whose plot_settings.txt matches the current
% settings. The file is deleted when a run starts and written after its last
% figure, so a folder from an interrupted run, or one drawn with different
% settings, is redrawn rather than kept. The key also fingerprints filt_neurons
% and, under the crop, the definitions file's timestamp. A redraw does not remove
% figures of slices it no longer draws.
%
% result fields:
%   out_dir     output folder
%   skipped     true when SKIP_EXISTING found a matching folder
%   dry_run     cfg.DRY_RUN
%   span_um     window width and height used, µm
%   um_per_pos  µm per filt_neurons.pos unit
%   slices      table, one row per figure written: slice, xlim, ylim, n_qc,
%               n_drawn, n_outside (cells on the figure outside the window)

% -- Settings check ----------------------------------------------------------
EXPECTED_FIELDS = {'MARKER', 'READS_THRESH', 'GENES_THRESH', 'MIN_ROLONIES', ...
    'DRAW_BELOW_CUTOFF', 'BELOW_COLOR', 'COLORMAP', 'CROP_TO_SUBSLICE', ...
    'SUBSLICE_DEFINITIONS_OVERRIDE', 'PANEL_COLUMNS', 'FIG_SIZE', 'MARKER_SIZE', ...
    'PNG_DPI', 'ANALYSIS_ROOT_OVERRIDE', 'AXIS_SPAN_UM', 'SKIP_EXISTING', ...
    'VERBOSE', 'DRY_RUN'};
assert(isstruct(cfg) && isscalar(cfg), 'cfg must be a scalar struct.');
missing = setdiff(EXPECTED_FIELDS, fieldnames(cfg));
unknown = setdiff(fieldnames(cfg), EXPECTED_FIELDS);
assert(isempty(missing), 'cfg is missing field(s): %s', strjoin(missing(:)', ', '));
assert(isempty(unknown), 'cfg has unknown field(s): %s', strjoin(unknown(:)', ', '));

MARKER                        = cfg.MARKER;
READS_THRESH                  = cfg.READS_THRESH;
GENES_THRESH                  = cfg.GENES_THRESH;
MIN_ROLONIES                  = cfg.MIN_ROLONIES;
DRAW_BELOW_CUTOFF             = cfg.DRAW_BELOW_CUTOFF;
BELOW_COLOR                   = cfg.BELOW_COLOR;
COLORMAP                      = cfg.COLORMAP;
CROP_TO_SUBSLICE              = cfg.CROP_TO_SUBSLICE;
SUBSLICE_DEFINITIONS_OVERRIDE = cfg.SUBSLICE_DEFINITIONS_OVERRIDE;
PANEL_COLUMNS                 = cfg.PANEL_COLUMNS;
FIG_SIZE                      = cfg.FIG_SIZE;
MARKER_SIZE                   = cfg.MARKER_SIZE;
PNG_DPI                       = cfg.PNG_DPI;
ANALYSIS_ROOT_OVERRIDE        = cfg.ANALYSIS_ROOT_OVERRIDE;
AXIS_SPAN_UM                  = cfg.AXIS_SPAN_UM;
VERBOSE                       = cfg.VERBOSE;

% Marker table -- the MATLAB counterpart of the project's marker_profiles.py.
% RAMP_MAX is the top of the count -> colour mapping and belongs here rather
% than in the config block: it DEFINES the mapping, so editing it re-shades
% every cell and two folders drawn at different RAMP_MAX are not comparable.
% It is in the output folder name so which one a figure used is never in doubt.
% Counts above it clamp to the top colour. RAMP_ANCHORS are marker_profiles.py's
% own anchors, used under COLORMAP = 'marker'. Columns are INDEX-ONLY: this
% panel labels its readout slots with stale gene names, so never resolve a
% marker by name.
% Anchors are dark -> mid -> bright, evenly spaced over the ramp domain, and
% interpolated linearly in RGB (what matplotlib's from_list does).
switch lower(MARKER)
    case 'mscarlet'
        MARKER_COLUMN  = 114;     % MATLAB 1-indexed (Python 113)
        MARKER_LABEL   = 'mScarlet';
        RAMP_ANCHORS   = [0.45 0.00 0.00; 1.00 0.35 0.00; 1.00 0.95 0.25];
        RAMP_MAX       = 15;      % BY95; marker_profiles.py's ceiling
    case 'gcamp'
        MARKER_COLUMN  = 112;     % MATLAB 1-indexed (Python 111)
        MARKER_LABEL   = 'GCaMP';
        RAMP_ANCHORS   = [0.00 0.42 0.10; 0.15 0.85 0.20; 0.80 1.00 0.40];
        RAMP_MAX       = 10;      % BY95; marker_profiles.py's ceiling
    otherwise
        error('MARKER must be ''mscarlet'' or ''gcamp'', got ''%s''.', MARKER);
end

assert(RAMP_MAX >= 1, 'RAMP_MAX must be at least 1.');
assert(MIN_ROLONIES >= 0, 'MIN_ROLONIES must be at least 0.');
if MIN_ROLONIES >= RAMP_MAX
    warning('MIN_ROLONIES (%g) is at or above RAMP_MAX (%g) -- every drawn cell saturates.', ...
        MIN_ROLONIES, RAMP_MAX);
end

assert(size(filt_neurons.expmat, 2) == PANEL_COLUMNS, ...
    ['expmat has %u columns but PANEL_COLUMNS says %u. Column %u is %s for a ' ...
     '%u-gene panel only; check this brain''s panel before plotting.'], ...
    size(filt_neurons.expmat, 2), PANEL_COLUMNS, MARKER_COLUMN, MARKER_LABEL, PANEL_COLUMNS);

project_root = fileparts(fileparts(mfilename('fullpath')));

% ANALYSIS_ROOT: read from local_config.py so it cannot drift from the value
% the Python pipeline uses. Line-anchored, so a commented-out line never wins.
if ~isempty(ANALYSIS_ROOT_OVERRIDE)
    analysis_root = ANALYSIS_ROOT_OVERRIDE;
else
    cfg_path = fullfile(project_root, 'local_config.py');
    assert(isfile(cfg_path), ...
        'local_config.py not found at %s -- set ANALYSIS_ROOT_OVERRIDE.', cfg_path);
    tok = regexp(fileread(cfg_path), ...
        '^\s*ANALYSIS_ROOT\s*=\s*r?[''"]([^''"]*)[''"]', ...
        'tokens', 'once', 'lineanchors');
    assert(~isempty(tok) && ~isempty(tok{1}), ...
        ['ANALYSIS_ROOT is unset in %s -- set it there, or set ' ...
         'ANALYSIS_ROOT_OVERRIDE in the config.'], cfg_path);
    analysis_root = tok{1};
end

% µm per pos unit, from scope_profiles.py by the same line-anchored read, so
% the figure cannot drift from the pixel size the Python pipeline uses.
scope_path = fullfile(project_root, 'scope_profiles.py');
assert(isfile(scope_path), 'scope_profiles.py not found at %s.', scope_path);
tok = regexp(fileread(scope_path), ...
    '^\s*EXVIVO_UM_PER_PX\s*=\s*([0-9.eE+-]+)', 'tokens', 'once', 'lineanchors');
assert(~isempty(tok), 'EXVIVO_UM_PER_PX not found in %s.', scope_path);
exvivo_um_per_px = str2double(tok{1});
assert(isfinite(exvivo_um_per_px) && exvivo_um_per_px > 0, ...
    'EXVIVO_UM_PER_PX in %s reads as %s, not a positive number.', scope_path, tok{1});
um_per_pos = 2 * exvivo_um_per_px;

if CROP_TO_SUBSLICE
    crop_dir = 'crop';
else
    crop_dir = 'full';
end
out_dir = fullfile(analysis_root, 'preprocessing', [MARKER_LABEL '_plots_dtc'], ...
    sprintf('qc%g_%g', READS_THRESH, GENES_THRESH), crop_dir, ...
    sprintf('ge%g_sat%g', MIN_ROLONIES, RAMP_MAX));
% The directory is created further down, after every check has passed -- making
% it here leaves an empty parameter folder behind when one of them raises.

% -- Frame -------------------------------------------------------------------
% From every row with a slice number: no QC, cutoff or crop, and nothing that
% depends on the marker, so every figure of a slice gets the same window.
% (:) on slice for the same reason as the masks below: a row-shaped field
% against a column implicitly expands to N-by-N.
slice_all = double(filt_neurons.slice(:));
assert(size(filt_neurons.pos, 2) >= 2 && size(filt_neurons.pos, 1) == numel(slice_all), ...
    'filt_neurons.pos is %s; expected %u rows (one per cell) of x, y.', ...
    mat2str(size(filt_neurons.pos)), numel(slice_all));
pos_um = double(filt_neurons.pos(:, 1:2)) * um_per_pos;   % N-by-2, x then y
has_slice = ~isnan(slice_all);
assert(any(has_slice), 'filt_neurons.slice holds no slice numbers.');
[grp, uniq_slices] = findgroups(slice_all(has_slice));
x_lo = splitapply(@min, pos_um(has_slice, 1), grp);
x_hi = splitapply(@max, pos_um(has_slice, 1), grp);
y_lo = splitapply(@min, pos_um(has_slice, 2), grp);
y_hi = splitapply(@max, pos_um(has_slice, 2), grp);
centre_x = (x_lo + x_hi) / 2;
centre_y = (y_lo + y_hi) / 2;
widest_um = max([x_hi - x_lo; y_hi - y_lo]);
if isempty(AXIS_SPAN_UM)
    span_um = widest_um;
    span_source = 'widest slice, all cells';
else
    span_um = AXIS_SPAN_UM;
    span_source = 'AXIS_SPAN_UM';
end
assert(isscalar(span_um) && span_um > 0, ...
    ['Window span is %s um (%s); it must be one positive number. Every slice ' ...
     'is a single point if the widest is 0 -- set AXIS_SPAN_UM.'], mat2str(span_um), span_source);
half = span_um / 2;

result = struct();
result.out_dir    = out_dir;
result.skipped    = false;
result.dry_run    = cfg.DRY_RUN;
result.span_um    = span_um;
result.um_per_pos = um_per_pos;
result.slices     = table(zeros(0, 1), zeros(0, 2), zeros(0, 2), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), 'VariableNames', ...
    {'slice', 'xlim', 'ylim', 'n_qc', 'n_drawn', 'n_outside'});

% Stepwise colormap: one row per integer count 0..RAMP_MAX, so with
% clim([-0.5 K+0.5]) count k lands in row k+1 -- discrete levels, no
% interpolation between counts. Row k+1 is the colour at frac = k/K.
K = round(RAMP_MAX);
ramp_frac = (0:K)' / K;
if strcmpi(COLORMAP, 'marker')
    CMAP = interp1(linspace(0, 1, size(RAMP_ANCHORS, 1)), RAMP_ANCHORS, ramp_frac, 'linear');
else
    assert(exist(COLORMAP, 'file') == 2 || exist(COLORMAP, 'builtin') == 5, ...
        'COLORMAP = ''%s'' is not a MATLAB colormap function. Try ''parula'' or ''marker''.', ...
        COLORMAP);
    CMAP = feval(COLORMAP, K + 1);
end
CMAP = min(max(CMAP, 0), 1);

if CROP_TO_SUBSLICE
    if ~isempty(SUBSLICE_DEFINITIONS_OVERRIDE)
        defs_path = SUBSLICE_DEFINITIONS_OVERRIDE;
    elseif strcmpi(MARKER, 'mscarlet')
        defs_path = fullfile(analysis_root, 'preprocessing', ...
            'subslice_definitions', 'subslice_definitions.mat');
    else
        defs_path = fullfile(analysis_root, 'preprocessing', ...
            'subslice_definitions', ['subslice_definitions_' lower(MARKER) '.mat']);
    end
    assert(isfile(defs_path), ...
        ['Subslice definitions not found:\n  %s\nRun\n  python ' ...
         'preprocessing/identify_marker_subslices.py --marker %s\n' ...
         'or set CROP_TO_SUBSLICE = false.'], defs_path, lower(MARKER));
end

% The settings text is the resume key, so it also fingerprints the inputs: a
% different filt_neurons, or a regenerated definitions file under the crop,
% does not match a folder drawn from the old one.
settings_path = fullfile(out_dir, 'plot_settings.txt');
derived = struct('MARKER_COLUMN', MARKER_COLUMN, 'RAMP_MAX', RAMP_MAX, ...
    'RAMP_ANCHORS', RAMP_ANCHORS, 'span_um', span_um, 'um_per_pos', um_per_pos, ...
    'data_cells', size(filt_neurons.expmat, 1), ...
    'data_nonzero', nnz(filt_neurons.expmat), ...
    'data_total_reads', full(sum(sum(filt_neurons.expmat))));
if CROP_TO_SUBSLICE
    defs_info = dir(defs_path);
    derived.subslice_definitions = defs_path;
    derived.subslice_definitions_modified = datestr(defs_info.datenum, 'yyyy-mm-dd HH:MM:SS');
end
settings_txt = settings_text(cfg, derived);

if cfg.DRY_RUN
    return
end
if cfg.SKIP_EXISTING && isfile(settings_path) && ...
        strcmp(strtrim(fileread(settings_path)), strtrim(settings_txt))
    result.skipped = true;
    vprintf(VERBOSE, 'skipped, settings match: %s\n', out_dir);
    return
end

countspercell = full(filt_neurons.expmat(:, MARKER_COLUMN));
total_cells   = numel(countspercell);
pass_qc = sum(filt_neurons.expmat, 2) >= READS_THRESH & ...
          sum(filt_neurons.expmat > 0, 2) >= GENES_THRESH;
total_passed  = nnz(pass_qc);

% Crop mask over every row: which cells sit in their slice's subslice FOVs.
% All true when cropping is off, so everything below reads the same either way.
in_crop = true(size(countspercell));
if CROP_TO_SUBSLICE
    defs = load(defs_path);   % path resolved and checked above
    assert(isfield(defs, 'subslice_info'), ...
        '%s holds no subslice_info struct.', defs_path);
    subslice_info = defs.subslice_info;

    % filt_neurons.fov is either the FOV names or numeric indices into
    % filt_neurons.fov_names -- utilities/mat_io.py resolves both, 1-based, and
    % the definitions file stores the resolved names.
    if isnumeric(filt_neurons.fov)
        assert(isfield(filt_neurons, 'fov_names'), ...
            'filt_neurons.fov is numeric but there is no fov_names to resolve it against.');
        fov_of_cell = filt_neurons.fov_names(filt_neurons.fov);
    else
        fov_of_cell = filt_neurons.fov;
    end
    assert(iscell(fov_of_cell), ...
        ['FOV names resolved to %s, not a cell array of names -- ismember would ' ...
         'compare the wrong thing rather than fail. Inspect filt_neurons.fov.'], ...
        class(fov_of_cell));

    in_crop = false(size(countspercell));
    for ii = 1:numel(subslice_info)
        % scipy.io.savemat writes a Python list of dicts as a 1xN CELL of 1x1
        % structs, not a struct array -- a list is not a mapping, so it takes
        % savemat's write_cells branch. Indexing it with () yields a 1x1 cell
        % and "Dot indexing is not supported". Handle both shapes so a file
        % written by MATLAB would also read.
        if iscell(subslice_info)
            entry = subslice_info{ii};
        else
            entry = subslice_info(ii);
        end

        % savemat turns the Python list of names into an N-by-L CHAR MATRIX,
        % not a cell of strings. `(:)` on that flattens column-major -- every
        % name's first character, then every name's second -- so ismember would
        % silently match nothing and every slice would look empty. cellstr
        % splits it back into rows and trims the padding, which is SPACES (
        % measured: savemat pads short names to the longest with char 32, not
        % NUL). deblank is belt and braces on top of that.
        fov_list = entry.fov_list;
        if ischar(fov_list)
            fov_list = cellstr(fov_list);
        end
        fov_list = deblank(fov_list(:));

        % (:) on both sides: a row-shaped slice field against a column mask
        % would implicitly expand into an N-by-N logical instead of erroring.
        in_slice_entry = filt_neurons.slice(:) == double(entry.slice_id);
        in_crop(in_slice_entry & ismember(fov_of_cell(:), fov_list)) = true;
    end
    vprintf(VERBOSE, 'crop: %s, %u slices, %u of %u QC-passing cells inside\n', ...
        defs_path, numel(subslice_info), nnz(pass_qc & in_crop), nnz(pass_qc));
end

vprintf(VERBOSE, '%s, column %u, reads >= %g, genes >= %g\n', ...
    MARKER_LABEL, MARKER_COLUMN, READS_THRESH, GENES_THRESH);
vprintf(VERBOSE, '  QC-passing cells:    %u / %u (%.1f%%)\n', ...
    total_passed, total_cells, total_passed / total_cells * 100);
vprintf(VERBOSE, '  median total reads:  %g\n', ...
    full(median(sum(filt_neurons.expmat(pass_qc, :), 2))));
vprintf(VERBOSE, '  mapping:             counts 0 .. %g+, %s, %u fixed levels\n', ...
    RAMP_MAX, lower(COLORMAP), K + 1);
vprintf(VERBOSE, '  drawn:               cells with >= %g rolonies\n', MIN_ROLONIES);
vprintf(VERBOSE, '  slices:              %u\n', numel(uniq_slices));
vprintf(VERBOSE, '  window:              %g um (%s), %.4g um per pos unit\n', ...
    span_um, span_source, um_per_pos);
vprintf(VERBOSE, '  output:              %s\n\n', out_dir);

slice_col   = zeros(0, 1);
counts      = zeros(0, 1);
n_passed    = zeros(0, 1);
n_drawn     = zeros(0, 1);
xlim_col    = zeros(0, 2);
ylim_col    = zeros(0, 2);
n_outside_c = zeros(0, 1);

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
% Gone until the last figure is written, so an interrupted run never leaves a
% folder that SKIP_EXISTING would take for finished.
if isfile(settings_path)
    delete(settings_path);
end
assert(~isfile(settings_path), ...
    'Could not delete %s; a later SKIP_EXISTING run could take this folder for finished.', ...
    settings_path);

for nn = 1:numel(uniq_slices)
    slice_no = uniq_slices(nn);
    sel   = slice_all == slice_no & pass_qc & in_crop;
    drawn = sel & countspercell >= MIN_ROLONIES;
    below = sel & ~drawn;

    % A slice with no subslice entry has nothing left after the crop. Skip it
    % rather than writing an empty figure -- step 1 drops slices with no
    % marker+ cells, so those slices are absent from the definitions file.
    if ~any(sel)
        if CROP_TO_SUBSLICE
            why = 'QC and crop';
        else
            why = 'QC';
        end
        vprintf(VERBOSE, 'slice %3u   no cells after %s -- skipped\n', slice_no, why);
        continue
    end

    med_count = full(median(sum(filt_neurons.expmat(sel, :), 2)));
    vprintf(VERBOSE, 'slice %3u   QC cells %6u   drawn %6u   median total reads %g\n', ...
        slice_no, nnz(sel), nnz(drawn), med_count);

    f = figure('Position', [50 50 FIG_SIZE], 'Visible', 'off');
    % Set CreateFcn AFTER creation: given at creation it runs immediately and
    % would undo 'Visible','off'. Set now it fires only when the .fig is
    % reopened, so a saved figure still opens visible without this run
    % throwing one window per slice.
    set(f, 'CreateFcn', 'set(gcbo,''Visible'',''on'')');
    ax = axes('Parent', f);
    hold(ax, 'on');

    if DRAW_BELOW_CUTOFF && any(below)
        scatter(ax, pos_um(below, 1), pos_um(below, 2), ...
            MARKER_SIZE, BELOW_COLOR, 'filled');
    end
    % Counts above RAMP_MAX clamp to it: they draw the top colour, which is
    % what that count maps to under the fixed span anyway. Drawn in ascending
    % count so the zero-count majority cannot cover a marker cell. Guarded
    % because a slice can have QC-passing cells and none at or above the
    % cutoff, and empty CData is the least exercised path through scatter.
    if any(drawn)
        idx = find(drawn);
        [~, order] = sort(countspercell(idx));
        idx = idx(order);
        scatter(ax, pos_um(idx, 1), pos_um(idx, 2), ...
            MARKER_SIZE, min(countspercell(idx), K), 'filled');
    end

    colormap(ax, CMAP);
    clim(ax, [-0.5, K + 0.5]);
    set(ax, 'ydir', 'reverse');

    xl = centre_x(nn) + [-half, half];
    yl = centre_y(nn) + [-half, half];
    axis(ax, 'image');
    xlim(ax, xl);
    ylim(ax, yl);
    xlabel(ax, 'x (\mum)');
    ylabel(ax, 'y (\mum)');

    % Tolerance: (lo+hi)/2 -/+ (hi-lo)/2 does not round back to lo and hi
    % exactly, so the widest slice's edge cells would otherwise count as outside
    % by ~1e-12 um.
    on_fig = drawn | (DRAW_BELOW_CUTOFF & below);
    fx = pos_um(on_fig, 1);
    fy = pos_um(on_fig, 2);
    tol = 1e-9 * span_um;
    n_outside = nnz(fx < xl(1) - tol | fx > xl(2) + tol | ...
                    fy < yl(1) - tol | fy > yl(2) + tol);
    if n_outside > 0
        warning('plot_marker_slices:clipped', ...
            'slice %u: %u cell(s) fall outside the %g um window and are not visible.', ...
            slice_no, n_outside, span_um);
    end

    % The colormap covers 0..K whatever the cutoff -- the mapping is built once
    % and never moves. The colorbar is cropped to what is actually on screen, so
    % raising MIN_ROLONIES shortens the legend from the bottom and leaves every
    % remaining swatch on the colour it already had.
    cb_lo = min(max(round(MIN_ROLONIES), 0), K);
    tick_step = max(1, ceil((K - cb_lo + 1) / 10));
    ticks = unique([cb_lo:tick_step:K, K]);
    cb = colorbar(ax);
    cb.Limits = [cb_lo - 0.5, K + 0.5];
    cb.Ticks = ticks;
    cb.TickLabels = [arrayfun(@num2str, ticks(1:end-1), 'UniformOutput', false), ...
                     {sprintf('%u+', K)}];
    cb.Label.String = 'rolonies';

    title(ax, sprintf('slice %u, %s  |  qc %g/%g  |  ge %g, ramp 0-%g+', ...
        slice_no, MARKER_LABEL, READS_THRESH, GENES_THRESH, MIN_ROLONIES, RAMP_MAX));

    % Named for the slice number, not the loop index: the two diverge as soon
    % as the slice numbering has a gap, and the original saved the index while
    % titling the slice. Zero-padded so the folder sorts in slice order.
    stem = fullfile(out_dir, sprintf('slice_%03u', slice_no));
    savefig(f, [stem '.fig']);
    exportgraphics(f, [stem '.png'], 'Resolution', PNG_DPI);
    close(f);

    slice_col   = [slice_col; slice_no];
    counts      = [counts; med_count];
    n_passed    = [n_passed; nnz(sel)];
    n_drawn     = [n_drawn; nnz(drawn)];
    xlim_col    = [xlim_col; xl];
    ylim_col    = [ylim_col; yl];
    n_outside_c = [n_outside_c; n_outside];
end

% Rows are the slices actually written, not every slice in the dataset -- a
% skipped slice has no figure and no row.
writetable( ...
    table(slice_col, n_passed, n_drawn, counts, 'VariableNames', ...
        {'slice', 'cells_passing_qc', 'cells_drawn', 'median_total_reads'}), ...
    fullfile(out_dir, 'median_total_reads.csv'));

fid = fopen(settings_path, 'w');
assert(fid > 0, 'Cannot write %s.', settings_path);
fprintf(fid, '%s\n', settings_txt);
fclose(fid);

result.slices = table(slice_col, xlim_col, ylim_col, n_passed, n_drawn, n_outside_c, ...
    'VariableNames', {'slice', 'xlim', 'ylim', 'n_qc', 'n_drawn', 'n_outside'});

vprintf(VERBOSE, '\n%u of %u slices written to %s\n', ...
    numel(slice_col), numel(uniq_slices), out_dir);
end


function txt = settings_text(cfg, derived)
% One "name = value" line per setting that changes what is written, sorted by
% name, then the values derived from them in the order given. SKIP_EXISTING,
% VERBOSE and DRY_RUN change how a run proceeds, not its output, so a folder
% drawn by gen_marker_plots_dtc.m still matches the sweep.
names = setdiff(fieldnames(cfg), {'SKIP_EXISTING', 'VERBOSE', 'DRY_RUN'});
lines = cell(numel(names), 1);
for ii = 1:numel(names)
    lines{ii} = sprintf('%s = %s', names{ii}, value_text(cfg.(names{ii})));
end
extra = fieldnames(derived);
for ii = 1:numel(extra)
    lines{end + 1} = sprintf('%s = %s', extra{ii}, value_text(derived.(extra{ii}))); %#ok<AGROW>
end
txt = strjoin(lines', newline);
end


function s = value_text(v)
if isstring(v)
    v = char(v);
end
if ischar(v)
    s = ['''' v ''''];
elseif isnumeric(v) || islogical(v)
    s = mat2str(v);
else
    error('Setting of class %s cannot be recorded in plot_settings.txt.', class(v));
end
end


function vprintf(verbose, varargin)
if verbose
    fprintf(varargin{:});
end
end
