% gen_marker_plots_dtc.m -- per-slice marker scatter, one marker per run.
%
% Replaces Gen_mScarlet_plots_dtc.m and Gen_GCaMP_plots_dtc.m: the marker is a
% config value below, not a separate file. Expects filt_neurons already in the
% workspace, as the originals do.
%
% Colour is stepwise and absolute. The count -> colour mapping is built first,
% over a span fixed per marker at [1, RAMP_MAX], one discrete level per integer
% count; MIN_ROLONIES then chooses which cells are displayed, and the colorbar
% is cropped to start at it. A 9-rolony cell draws the same colour at a cutoff
% of 1 and at a cutoff of 5, and the ramp's slope never changes. COLORMAP picks the palette: any MATLAB
% colormap name (parula is the blue -> yellow default), or 'marker' for the
% marker's own ramp out of the project's marker_profiles.py, which is what the
% cellmask renders paint with.
%
% Writes to
%   <ANALYSIS_ROOT>\preprocessing\<Marker>_plots_dtc\qc<reads>_<genes>_ge<cut>_sat<cap>_<cmap>\
% one .fig and one .png per slice, plus median_counts.csv. The parameters are
% in the folder name, so changing one writes a new folder rather than
% overwriting the previous run.

%% ---- CONFIG ---------------------------------------------------------------
% -- Marker ----------------------------------------------------------------
MARKER        = 'mscarlet';   % 'mscarlet' | 'gcamp'

% -- QC --------------------------------------------------------------------
% QC floors for a cell to be plotted at all. Deliberately NOT inherited from
% local_config.py's QC_MIN_READS / QC_MIN_GENES -- those gate the alignment
% pipeline, these gate a figure, and the lab's marker plots run 0/0 so marker
% detection is not gated on transcriptome quality. Per-brain either way: read
% the dataset's own Gen_*_plots.m rather than carrying another brain's numbers.
READS_THRESH  = 20;
GENES_THRESH  = 5;

% -- Rolony cutoff and colour ----------------------------------------------
% Rolony cutoff: a cell below this is not painted. 1 draws every marker+ cell.
% The marker's step-4 draw floor is 5 (mScarlet) / 3 (GCaMP) if you want this
% figure to match what the pipeline renders. Changing it does NOT change any
% remaining cell's colour.
MIN_ROLONIES  = 1;

% Cells below MIN_ROLONIES as a flat grey, for when the section outline is
% wanted behind the marker cells. Off by default -- a QC-passing cell with 0
% rolonies carries no marker signal and is not what these figures are for.
% 0.25 is the grey the pipeline's cellmask field paints at.
DRAW_BELOW_CUTOFF = false;
BELOW_COLOR       = [0.25 0.25 0.25];

% The span of the count -> colour mapping is NOT here on purpose: it is
% RAMP_MAX in the marker table below, a fixed per-marker constant. Its slope is
% (top colour - bottom colour) / (RAMP_MAX - 1), so a per-run dial would change
% the slope between runs -- exactly what the absolute ramp exists to prevent.

% Palette. Any MATLAB colormap function name -- 'parula' (blue -> yellow),
% 'turbo', 'hot', 'jet' -- or 'marker' for this marker's own dark-to-bright
% ramp from marker_profiles.py, which makes a count draw the same colour here as
% in the cellmask renders. Either way it is sampled into RAMP_MAX discrete
% levels, one per count.
COLORMAP      = 'parula';

% -- Guards ----------------------------------------------------------------
% Column count of this brain's panel. Guard only: a panel with at least
% MARKER_COLUMN columns but a different slot order would otherwise plot the
% wrong gene with no error.
PANEL_COLUMNS = 114;

% -- Figure ----------------------------------------------------------------
FIG_SIZE      = [600 600];    % pixels, [width height]
SQUARE_AXES   = true;         % x and y over one range, equal unit length

% One extent for every slice, taken from the widest QC-passing section, so two
% figures are at the same scale and comparable by eye. false frames each slice
% on its own cells, which fills the box but makes the scale differ per figure.
COMMON_EXTENT = true;
MARKER_SIZE   = 5;            % scatter point area
PNG_DPI       = 300;

% -- Output ----------------------------------------------------------------
% Blank -> read ANALYSIS_ROOT from local_config.py one level up. Set a path
% here to write somewhere else instead.
ANALYSIS_ROOT_OVERRIDE = '';
%% ---------------------------------------------------------------------------

% Marker table -- the MATLAB counterpart of the project's marker_profiles.py.
% RAMP_MAX is the top of the count -> colour mapping and belongs here rather
% than in the config block: it DEFINES the mapping, so editing it re-shades
% every cell and two folders drawn at different RAMP_MAX are not comparable.
% It is in the output folder name so which one a figure used is never in doubt.
% Counts above it clamp to the top colour.
% and the anchors are that file's, so a count renders the same colour here as
% in the cellmask renders. Columns are INDEX-ONLY: this panel labels its
% readout slots with stale gene names, so never resolve a marker by name.
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
if MIN_ROLONIES >= RAMP_MAX
    warning('MIN_ROLONIES (%g) is at or above RAMP_MAX (%g) -- every drawn cell saturates.', ...
        MIN_ROLONIES, RAMP_MAX);
end

assert(exist('filt_neurons', 'var') == 1, ...
    'filt_neurons is not in the workspace -- load the brain''s filt_neurons.mat first.');
assert(size(filt_neurons.expmat, 2) == PANEL_COLUMNS, ...
    ['expmat has %u columns but PANEL_COLUMNS says %u. Column %u is %s for a ' ...
     '%u-gene panel only; check this brain''s panel before plotting.'], ...
    size(filt_neurons.expmat, 2), PANEL_COLUMNS, MARKER_COLUMN, MARKER_LABEL, PANEL_COLUMNS);

% ANALYSIS_ROOT: read from local_config.py so it cannot drift from the value
% the Python pipeline uses. Line-anchored, so a commented-out line never wins.
if ~isempty(ANALYSIS_ROOT_OVERRIDE)
    analysis_root = ANALYSIS_ROOT_OVERRIDE;
else
    cfg_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'local_config.py');
    assert(isfile(cfg_path), ...
        'local_config.py not found at %s -- set ANALYSIS_ROOT_OVERRIDE.', cfg_path);
    tok = regexp(fileread(cfg_path), ...
        '^\s*ANALYSIS_ROOT\s*=\s*r?[''"]([^''"]*)[''"]', ...
        'tokens', 'once', 'lineanchors');
    assert(~isempty(tok) && ~isempty(tok{1}), ...
        ['ANALYSIS_ROOT is unset in %s -- set it there, or set ' ...
         'ANALYSIS_ROOT_OVERRIDE above.'], cfg_path);
    analysis_root = tok{1};
end

param_dir = sprintf('qc%g_%g_ge%g_sat%g_%s', ...
    READS_THRESH, GENES_THRESH, MIN_ROLONIES, RAMP_MAX, lower(COLORMAP));
out_dir = fullfile(analysis_root, 'preprocessing', [MARKER_LABEL '_plots_dtc'], param_dir);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% Stepwise colormap: one row per integer count 1..RAMP_MAX, so with
% clim([0.5 K+0.5]) a count lands in its own row -- discrete levels, no
% interpolation between counts. Row k is the colour of count k at
% frac = (k-1)/(K-1), the mapping check_rolony_cutoff.py uses, so under
% COLORMAP = 'marker' a count draws the same colour here as in the cellmask
% renders.
K = round(RAMP_MAX);
ramp_frac = ((1:K)' - 1) / max(K - 1, 1);
if strcmpi(COLORMAP, 'marker')
    CMAP = interp1(linspace(0, 1, size(RAMP_ANCHORS, 1)), RAMP_ANCHORS, ramp_frac, 'linear');
else
    assert(exist(COLORMAP, 'file') == 2 || exist(COLORMAP, 'builtin') == 5, ...
        'COLORMAP = ''%s'' is not a MATLAB colormap function. Try ''parula'' or ''marker''.', ...
        COLORMAP);
    CMAP = feval(COLORMAP, K);
end
CMAP = min(max(CMAP, 0), 1);

uniq_slices = unique(filt_neurons.slice);
uniq_slices = uniq_slices(~isnan(uniq_slices));

countspercell = full(filt_neurons.expmat(:, MARKER_COLUMN));
total_cells   = numel(countspercell);
pass_qc = sum(filt_neurons.expmat, 2) >= READS_THRESH & ...
          sum(filt_neurons.expmat > 0, 2) >= GENES_THRESH;
total_passed  = nnz(pass_qc);

fprintf('%s, column %u, reads >= %g, genes >= %g\n', ...
    MARKER_LABEL, MARKER_COLUMN, READS_THRESH, GENES_THRESH);
fprintf('  QC-passing cells:    %u / %u (%.1f%%)\n', ...
    total_passed, total_cells, total_passed / total_cells * 100);
fprintf('  median total counts: %g\n', ...
    full(median(sum(filt_neurons.expmat(pass_qc, :), 2))));
fprintf('  mapping:             counts 1 .. %g+, %s, %u fixed levels\n', ...
    RAMP_MAX, lower(COLORMAP), K);
fprintf('  drawn:               cells with >= %g rolonies\n', MIN_ROLONIES);
fprintf('  slices:              %u\n', numel(uniq_slices));
fprintf('  output:              %s\n\n', out_dir);

% Widest QC-passing section, so every figure can be drawn at one scale. Taken
% from the QC-passing population, not the drawn one, so MIN_ROLONIES cannot
% change the frame.
common_span = 0;
if COMMON_EXTENT
    for ii = 1:numel(uniq_slices)
        span_sel = filt_neurons.slice == uniq_slices(ii) & pass_qc;
        if ~any(span_sel)
            continue
        end
        sx = filt_neurons.pos(span_sel, 1);
        sy = filt_neurons.pos(span_sel, 2);
        common_span = max([common_span, max(sx) - min(sx), max(sy) - min(sy)]);
    end
    fprintf('  common extent:       %g pos units, same in every slice\n', common_span);
end

counts   = [];
n_passed = [];
n_drawn  = [];

for nn = 1:numel(uniq_slices)
    slice_no = uniq_slices(nn);
    sel   = filt_neurons.slice == slice_no & pass_qc;
    drawn = sel & countspercell >= MIN_ROLONIES;
    below = sel & ~drawn;

    med_count = full(median(sum(filt_neurons.expmat(sel, :), 2)));
    fprintf('slice %3u   QC cells %6u   drawn %6u   median count %g\n', ...
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
        scatter(ax, filt_neurons.pos(below, 1), filt_neurons.pos(below, 2), ...
            MARKER_SIZE, BELOW_COLOR, 'filled');
    end
    % Counts above RAMP_MAX clamp to it: they draw the top colour, which is
    % what that count maps to under the fixed span anyway.
    scatter(ax, filt_neurons.pos(drawn, 1), filt_neurons.pos(drawn, 2), ...
        MARKER_SIZE, min(countspercell(drawn), K), 'filled');

    colormap(ax, CMAP);
    clim(ax, [0.5, K + 0.5]);
    set(ax, 'ydir', 'reverse');

    % Frame from every QC-passing cell in the slice, not from the drawn subset:
    % taking it from the plotted data would let MIN_ROLONIES zoom the section,
    % and the cutoff must change which cells are drawn and nothing else.
    px = filt_neurons.pos(sel, 1);
    py = filt_neurons.pos(sel, 2);
    if SQUARE_AXES
        axis(ax, 'image');
    end
    if ~isempty(px)
        if COMMON_EXTENT && common_span > 0
            % One span for every slice, centred on this slice's own cells, so
            % two figures are at the same scale whatever each section measures.
            half = common_span / 2;
            xlim(ax, (max(px) + min(px)) / 2 + [-half half]);
            ylim(ax, (max(py) + min(py)) / 2 + [-half half]);
        elseif SQUARE_AXES
            lims = [min([px; py]), max([px; py])];
            xlim(ax, lims); ylim(ax, lims);
        else
            xlim(ax, [min(px) max(px)]);
            ylim(ax, [min(py) max(py)]);
        end
    end

    % The colormap covers 1..K whatever the cutoff -- the mapping is built once
    % and never moves. The colorbar is cropped to what is actually on screen, so
    % raising MIN_ROLONIES shortens the legend from the bottom and leaves every
    % remaining swatch on the colour it already had.
    cb_lo = min(max(round(MIN_ROLONIES), 1), K);
    tick_step = max(1, ceil((K - cb_lo + 1) / 10));
    ticks = unique([cb_lo:tick_step:K, K]);
    cb = colorbar(ax);
    cb.Limits = [cb_lo - 0.5, K + 0.5];
    cb.Ticks = ticks;
    cb.TickLabels = [arrayfun(@num2str, ticks(1:end-1), 'UniformOutput', false), ...
                     {sprintf('%u+', K)}];
    cb.Label.String = 'rolonies';

    title(ax, sprintf('slice %u, %s  |  ge %g, ramp 1-%g+', ...
        slice_no, MARKER_LABEL, MIN_ROLONIES, RAMP_MAX));

    % Named for the slice number, not the loop index: the two diverge as soon
    % as the slice numbering has a gap, and the original saved the index while
    % titling the slice. Zero-padded so the folder sorts in slice order.
    stem = fullfile(out_dir, sprintf('slice_%03u', slice_no));
    savefig(f, [stem '.fig']);
    exportgraphics(f, [stem '.png'], 'Resolution', PNG_DPI);
    close(f);

    counts   = [counts; med_count];
    n_passed = [n_passed; nnz(sel)];
    n_drawn  = [n_drawn; nnz(drawn)];
end

writetable( ...
    table(uniq_slices(:), n_passed, n_drawn, counts, 'VariableNames', ...
        {'slice', 'cells_passing_qc', 'cells_drawn', 'median_total_counts'}), ...
    fullfile(out_dir, 'median_counts.csv'));

fprintf('\n%u slices written to %s\n', numel(uniq_slices), out_dir);
