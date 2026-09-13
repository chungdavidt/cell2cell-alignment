% test_plot_marker_slices.m -- check plot_marker_slices.m's logic against answers
% worked out by hand, on a synthetic filt_neurons small enough to count.
%
%   test_plot_marker_slices                 % synthetic checks only, seconds
%   test_plot_marker_slices(filt_neurons)   % plus one real run, ~1 min
%
% Run from matlab\ (or with it on the path). Writes only under tempdir and
% prints the folder. Anything that reports FAIL is a defect in
% plot_marker_slices.m, not in this file -- every expected number below is
% derived in the comments from the cell table, never by calling the code under
% test.

function test_plot_marker_slices(real_filt_neurons)
if nargin < 1, real_filt_neurons = []; end
fprintf('MATLAB %s\n\n', version);
n_fail = 0;
root = fullfile(tempdir, ['test_plot_marker_slices_' datestr(now, 'yyyymmdd_HHMMSS')]);
fprintf('writing under %s\n\n', root);

% -- The synthetic brain -----------------------------------------------------
% One row per cell: slice, pos x, pos y (pos units), total reads, genes
% detected, mScarlet count (col 114), GCaMP count (col 112). make_row builds an
% expmat row with exactly these totals.
%
%   cell  slice   x     y    reads genes mSc GCaMP   20/5?
%   c1      1    100   200    30    6     0    2     pass
%   c2      1    300   250    25    5     3    0     pass
%   c3      1    200   400    40    8    20   12     pass
%   c4      1    250   300    10    3     1    1     fail
%   c5      1   2100   200     2    1     0    0     fail   stray, far right
%   c6      2   5000  5000    50   10     5    0     pass
%   c7      2   5600  5300    20    5     0    4     pass   on both floors
%   c8      2   5300  6000    19    5     2    0     fail   reads 19
%   c9      2   5100  5800    20    4     1    1     fail   genes 4
%   c10     3   9000   100     5    2     4    0     fail
%   c11     3   9400   900     8    3     0    3     fail
%   c12   NaN  50000 50000   100   20    50    0     -      no slice: ignored
CELLS = [ ...
    1   100   200   30  6   0  2
    1   300   250   25  5   3  0
    1   200   400   40  8  20 12
    1   250   300   10  3   1  1
    1  2100   200    2  1   0  0
    2  5000  5000   50 10   5  0
    2  5600  5300   20  5   0  4
    2  5300  6000   19  5   2  0
    2  5100  5800   20  4   1  1
    3  9000   100    5  2   4  0
    3  9400   900    8  3   0  3
  NaN 50000 50000  100 20  50  0];
fn = struct();
fn.slice = CELLS(:, 1);
fn.pos = CELLS(:, 2:3);
expmat = zeros(size(CELLS, 1), 114);
for ii = 1:size(CELLS, 1)
    expmat(ii, :) = make_row(CELLS(ii, 4), CELLS(ii, 5), CELLS(ii, 6), CELLS(ii, 7));
end
fn.expmat = sparse(expmat);

UM = 0.64;   % 2 * EXVIVO_UM_PER_PX = 2 * 0.32
% Bounding boxes over ALL cells of a slice, pos units:
%   slice 1: x 100..2100 (c5 included), y 200..400  -> centre (1100, 300)
%   slice 2: x 5000..5600,              y 5000..6000 -> centre (5300, 5500)
%   slice 3: x 9000..9400,              y 100..900   -> centre (9200, 500)
% Widest side: slice 1's x, 2000 pos = 1280 um. Without c5 it would be slice 2's
% y, 1000 pos = 640 um; with c12 counted it would be ~46000 pos.
SPAN = 1280;
CENTRE = [1100 300; 5300 5500; 9200 500] * UM;

base = struct();
base.MARKER = 'mscarlet';
base.READS_THRESH = 20;
base.GENES_THRESH = 5;
base.MIN_ROLONIES = 0;
base.ROLONY_CEILING = [];
base.RAMP_FROM = 'floor';
base.DRAW_BELOW_CUTOFF = false;
base.BELOW_COLOR = [0.25 0.25 0.25];
base.COLORMAP = 'parula';
base.CROP_TO_SUBSLICE = false;
base.SUBSLICE_DEFINITIONS_OVERRIDE = '';
base.PANEL_COLUMNS = 114;
base.FIG_SIZE = [300 300];
base.MARKER_SIZE = 5;
base.PNG_DPI = 50;
base.ANALYSIS_ROOT_OVERRIDE = root;
base.AXIS_SPAN_UM = [];
base.SKIP_EXISTING = false;
base.VERBOSE = false;
base.DRY_RUN = false;

% -- 1. output path, and a dry run writes nothing --------------------------
c = base; c.MIN_ROLONIES = 2; c.DRY_RUN = true;
r = plot_marker_slices(fn, c);
want = fullfile(root, 'preprocessing', 'mScarlet_plots_dtc', 'qc20_5', 'full', 'ge2_sat15');
n_fail = report(strcmp(r.out_dir, want) && ~isfolder(root), ...
    '1a. mScarlet path is qc20_5\full\ge2_sat15 and the dry run created nothing', ...
    sprintf('got %s, root exists = %d', r.out_dir, isfolder(root)), n_fail);
c.MARKER = 'gcamp'; c.READS_THRESH = 0; c.GENES_THRESH = 0; c.MIN_ROLONIES = 3;
r = plot_marker_slices(fn, c);
want = fullfile(root, 'preprocessing', 'GCaMP_plots_dtc', 'qc0_0', 'full', 'ge3_sat10');
n_fail = report(strcmp(r.out_dir, want), '1b. GCaMP path is qc0_0\full\ge3_sat10', ...
    sprintf('got %s', r.out_dir), n_fail);

% -- 2/3. frame: same window for every QC pair, cutoff and marker; right values
runs = {};
frame_ok = true;
detail = '';
for marker = {'mscarlet', 'gcamp'}
    for qc = [20 5; 0 0]'
        for cut = [0 3]
            c = base; c.MARKER = marker{1};
            c.READS_THRESH = qc(1); c.GENES_THRESH = qc(2); c.MIN_ROLONIES = cut;
            r = plot_marker_slices(fn, c);
            runs{end + 1} = r; %#ok<AGROW>
            for jj = 1:height(r.slices)
                s = r.slices.slice(jj);
                wx = CENTRE(s, 1) + [-SPAN SPAN] / 2;
                wy = CENTRE(s, 2) + [-SPAN SPAN] / 2;
                if max(abs(r.slices.xlim(jj, :) - wx)) > 1e-9 || ...
                        max(abs(r.slices.ylim(jj, :) - wy)) > 1e-9
                    frame_ok = false;
                    detail = sprintf('%s qc%d_%d ge%d slice %d: xlim %s want %s, ylim %s want %s', ...
                        marker{1}, qc(1), qc(2), cut, s, mat2str(r.slices.xlim(jj, :)), ...
                        mat2str(wx), mat2str(r.slices.ylim(jj, :)), mat2str(wy));
                end
            end
        end
    end
end
n_fail = report(frame_ok, ...
    '2a. every slice has the hand-computed window in all 8 runs (2 markers x 2 QC x 2 cutoffs)', ...
    detail, n_fail);
% The same, read back from the saved figures: slice 1 is drawn in all 8 runs.
fig_ok = true;
detail = '';
wx = CENTRE(1, 1) + [-SPAN SPAN] / 2;
wy = CENTRE(1, 2) + [-SPAN SPAN] / 2;
for jj = 1:numel(runs)
    [~, ~, ~, fxl, fyl] = read_points(fullfile(runs{jj}.out_dir, 'slice_001.fig'));
    if max(abs(fxl - wx)) > 1e-9 || max(abs(fyl - wy)) > 1e-9
        fig_ok = false;
        detail = sprintf('%s: saved XLim %s YLim %s, want %s %s', runs{jj}.out_dir, ...
            mat2str(fxl), mat2str(fyl), mat2str(wx), mat2str(wy));
    end
end
n_fail = report(fig_ok, '2b. the saved .fig of slice 1 has that window in all 8 runs', ...
    detail, n_fail);
spans = cellfun(@(x) x.span_um, runs);
n_fail = report(all(abs(spans - SPAN) < 1e-9), ...
    '3a. span is 1280 um: slice 1 widened by the stray cell, the NaN-slice cell ignored', ...
    sprintf('spans %s', mat2str(spans)), n_fail);
n_fail = report(abs(runs{1}.um_per_pos - UM) < 1e-12, ...
    '3b. um_per_pos read from scope_profiles.py is 0.64', ...
    sprintf('got %g', runs{1}.um_per_pos), n_fail);

% -- 4. AXIS_SPAN_UM overrides the span, and clipping is counted ------------
% 0/0, cutoff 0: all five slice-1 cells are on the figure. The window is
% x 704 +- 50 um; the cells' x are 64, 192, 128, 160, 1344 um, all outside.
c = base; c.READS_THRESH = 0; c.GENES_THRESH = 0; c.AXIS_SPAN_UM = 100;
c.ANALYSIS_ROOT_OVERRIDE = fullfile(root, 'override');
warning('off', 'plot_marker_slices:clipped');
r = plot_marker_slices(fn, c);
warning('on', 'plot_marker_slices:clipped');
row1 = r.slices.slice == 1;
n_fail = report(abs(diff(r.slices.xlim(row1, :)) - 100) < 1e-9 && ...
    abs(diff(r.slices.ylim(row1, :)) - 100) < 1e-9 && r.span_um == 100, ...
    '4a. AXIS_SPAN_UM = 100 sets a 100 um window', ...
    sprintf('xlim %s', mat2str(r.slices.xlim(row1, :))), n_fail);
n_fail = report(r.slices.n_outside(row1) == 5, ...
    '4b. all 5 slice-1 cells are counted outside the 100 um window', ...
    sprintf('n_outside = %d', r.slices.n_outside(row1)), n_fail);
n_fail = report(all(runs{1}.slices.n_outside == 0), ...
    '4c. with the measured span no cell is outside', ...
    sprintf('n_outside %s', mat2str(runs{1}.slices.n_outside)), n_fail);

% -- 5-8. what is drawn, in what colour and order, in um ----------------------
% runs{1} = mscarlet 20/5 ge0; runs{2} = mscarlet 20/5 ge3;
% runs{3} = mscarlet 0/0 ge0;  runs{4} = mscarlet 0/0 ge3.
% 20/5 ge0, slice 1: c1 c2 c3 pass -> 3 points.
[xd, ~, cd] = read_points(fullfile(runs{1}.out_dir, 'slice_001.fig'));
n_fail = report(numel(xd) == 3, '5a. 20/5 ge0 slice 1 draws the 3 QC-passing cells', ...
    sprintf('%d points', numel(xd)), n_fail);
% 0/0 ge3, slice 1: mScarlet c2 = 3, c3 = 20 -> 2 points, CData [3 15].
[xd, ~, cd] = read_points(fullfile(runs{4}.out_dir, 'slice_001.fig'));
n_fail = report(numel(xd) == 2 && isequal(cd(:)', [3 15]), ...
    '5b/6. 0/0 ge3 slice 1 draws c2 and c3 only, CData [3 15] (20 clamps to 15)', ...
    sprintf('%d points, CData %s', numel(xd), mat2str(cd(:)')), n_fail);
% 0/0 ge0, slice 1: counts in row order c1..c5 = [0 3 20 1 0]. A stable
% ascending sort gives c1 c5 c4 c2 c3 -> CData [0 0 1 3 15],
% x = [100 2100 250 300 200] pos, y = [200 200 300 250 400] pos.
[xd, yd, cd] = read_points(fullfile(runs{3}.out_dir, 'slice_001.fig'));
n_fail = report(isequal(cd(:)', [0 0 1 3 15]), ...
    '7. cells are drawn in ascending count: CData [0 0 1 3 15]', ...
    sprintf('CData %s', mat2str(cd(:)')), n_fail);
n_fail = report(numel(xd) == 5 && max(abs(xd(:)' - [100 2100 250 300 200] * UM)) < 1e-9 && ...
    max(abs(yd(:)' - [200 200 300 250 400] * UM)) < 1e-9, ...
    '8. plotted x/y are pos * 0.64, in the draw order', ...
    sprintf('x %s', mat2str(xd(:)')), n_fail);
% GCaMP 20/5 ge3, slice 2: c6 GCaMP 0, c7 GCaMP 4 -> 1 point, CData 4.
[xd, ~, cd] = read_points(fullfile(runs{6}.out_dir, 'slice_002.fig'));
n_fail = report(numel(xd) == 1 && isequal(cd, 4), ...
    '5c. GCaMP 20/5 ge3 slice 2 draws only c7 (col 112 = 4)', ...
    sprintf('%d points, CData %s', numel(xd), mat2str(cd)), n_fail);

% Title and labels.
g = openfig(fullfile(runs{2}.out_dir, 'slice_001.fig'), 'invisible');
ax = findobj(g, 'Type', 'axes');
ttl = get(get(ax(1), 'Title'), 'String');
xlab = get(get(ax(1), 'XLabel'), 'String');
close(g);
n_fail = report(contains(ttl, 'qc 20/5') && contains(ttl, 'ge 3') && strcmp(xlab, 'x (\mum)'), ...
    '8b. title carries the QC pair and cutoff; x label is um', ...
    sprintf('title "%s", xlabel "%s"', ttl, xlab), n_fail);

% -- 11. a slice with no QC-passing cells gets no figure and no CSV row ------
% 20/5 ge3 mScarlet: slice 1 c2 c3 -> 2 drawn of 3 QC; slice 2 c6 -> 1 of 2;
% slice 3 nothing passes. Median total reads: slice 1 median(30,25,40) = 30,
% slice 2 median(50,20) = 35.
T = readtable(fullfile(runs{2}.out_dir, 'median_total_reads.csv'));
ok = isequal(T.slice(:)', [1 2]) && isequal(T.cells_passing_qc(:)', [3 2]) && ...
    isequal(T.cells_drawn(:)', [2 1]) && isequal(T.median_total_reads(:)', [30 35]) && ...
    ~isfile(fullfile(runs{2}.out_dir, 'slice_003.fig')) && ...
    ~isfile(fullfile(runs{2}.out_dir, 'slice_003.png'));
n_fail = report(ok, ...
    '11. 20/5 ge3 CSV = slices [1 2], QC [3 2], drawn [2 1], median [30 35]; no slice 3 files', ...
    sprintf('slice %s qc %s drawn %s median %s', mat2str(T.slice(:)'), ...
    mat2str(T.cells_passing_qc(:)'), mat2str(T.cells_drawn(:)'), ...
    mat2str(T.median_total_reads(:)')), n_fail);
% 0/0 ge3: slice 3 c10 mScarlet 4 passes -> it now has a figure.
n_fail = report(isfile(fullfile(runs{4}.out_dir, 'slice_003.png')), ...
    '11b. at 0/0 slice 3 is drawn', '', n_fail);

% -- 9. resume ---------------------------------------------------------------
png = fullfile(runs{1}.out_dir, 'slice_001.png');
before = dir(png);
pause(1.1);
c = base; c.SKIP_EXISTING = true;
r = plot_marker_slices(fn, c);
after = dir(png);
n_fail = report(r.skipped && after.datenum == before.datenum, ...
    '9a. identical settings with SKIP_EXISTING: skipped, PNG untouched', ...
    sprintf('skipped = %d, PNG time changed = %d', r.skipped, after.datenum ~= before.datenum), n_fail);
c.AXIS_SPAN_UM = 2000;
r = plot_marker_slices(fn, c);
n_fail = report(~r.skipped, '9b. a changed AXIS_SPAN_UM is redrawn, not skipped', ...
    sprintf('skipped = %d', r.skipped), n_fail);
c.AXIS_SPAN_UM = [];
c.VERBOSE = true;
evalc('r = plot_marker_slices(fn, c);');
n_fail = report(~r.skipped, '9c. switching back is redrawn too (the file holds the last settings)', ...
    sprintf('skipped = %d', r.skipped), n_fail);
c.VERBOSE = false;   % the only difference from the run that wrote the file
r = plot_marker_slices(fn, c);
n_fail = report(r.skipped, '9d. VERBOSE does not count as a setting: skipped', ...
    sprintf('skipped = %d', r.skipped), n_fail);
delete(fullfile(r.out_dir, 'plot_settings.txt'));
r = plot_marker_slices(fn, c);
n_fail = report(~r.skipped && isfile(fullfile(r.out_dir, 'plot_settings.txt')), ...
    '9e. a folder with no plot_settings.txt (interrupted run) is redrawn', ...
    sprintf('skipped = %d', r.skipped), n_fail);

% -- 10. settings check --------------------------------------------------------
c = rmfield(base, 'VERBOSE');
msg = error_message(@() plot_marker_slices(fn, c));
n_fail = report(contains(msg, 'missing') && contains(msg, 'VERBOSE'), ...
    '10a. a missing field errors and names it', sprintf('message: %s', msg), n_fail);
c = base; c.READ_THRESH = 20;
msg = error_message(@() plot_marker_slices(fn, c));
n_fail = report(contains(msg, 'unknown') && contains(msg, 'READ_THRESH'), ...
    '10b. an unknown field errors and names it', sprintf('message: %s', msg), n_fail);

% -- 12. ceiling and ramp start -------------------------------------------------
% Slice 1 at QC 0/0, mScarlet; its counts in row order c1..c5 are [0 3 20 1 0].
% 12a. ceiling 5, floor 0, 'floor': drawn in ascending order [0 0 1 3 20],
%      clamped to [0 0 1 3 5]; 6 levels 0..5 -> CLim [-0.5 5.5], parula(6);
%      colorbar 0 1 2 3 4 5+.
c = base; c.READS_THRESH = 0; c.GENES_THRESH = 0; c.ROLONY_CEILING = 5;
r = plot_marker_slices(fn, c);
[~, ~, cd, ~, ~, cl, cm, cbl, cbt] = read_points(fullfile(r.out_dir, 'slice_001.fig'));
want = fullfile(root, 'preprocessing', 'mScarlet_plots_dtc', 'qc0_0', 'full', 'ge0_sat5');
ok = strcmp(r.out_dir, want) && isequal(cd(:)', [0 0 1 3 5]) && ...
    max(abs(cl - [-0.5 5.5])) < 1e-12 && isequal(size(cm), [6 3]) && ...
    max(abs(cm - parula(6)), [], 'all') < 1e-12 && ...
    max(abs(cbl - [-0.5 5.5])) < 1e-12 && isequal(cbt(:)', {'0', '1', '2', '3', '4', '5+'});
n_fail = report(ok, '12a. ceiling 5, floor 0: ge0_sat5, CData [0 0 1 3 5], CLim [-0.5 5.5], parula(6), bar 0..5+', ...
    sprintf('%s CData %s CLim %s rows %d bar %s', r.out_dir, mat2str(cd(:)'), mat2str(cl), ...
    size(cm, 1), strjoin(cbt(:)', ' ')), n_fail);
% 12b. ceiling 5, floor 3, 'floor': drawn c2 (3), c3 (20) -> CData [3 5];
%      3 levels 3..5 -> CLim [2.5 5.5], parula(3); full-height bar 3 4 5+.
c.MIN_ROLONIES = 3;
r = plot_marker_slices(fn, c);
[~, ~, cd, ~, ~, cl, cm, cbl, cbt, ttl, cbk] = read_points(fullfile(r.out_dir, 'slice_001.fig'));
want = fullfile(root, 'preprocessing', 'mScarlet_plots_dtc', 'qc0_0', 'full', 'ge3_sat5');
ok = strcmp(r.out_dir, want) && isequal(cd(:)', [3 5]) && ...
    max(abs(cl - [2.5 5.5])) < 1e-12 && isequal(size(cm), [3 3]) && ...
    max(abs(cm - parula(3)), [], 'all') < 1e-12 && ...
    max(abs(cbl - [2.5 5.5])) < 1e-12 && isequal(cbt(:)', {'3', '4', '5+'}) && ...
    isequal(cbk(:)', [3 4 5]) && contains(ttl, 'colours 3-5+');
n_fail = report(ok, '12b. floor 3, ceiling 5, ramp from floor: ge3_sat5, CLim [2.5 5.5], parula(3), bar 3..5+', ...
    sprintf('%s CData %s CLim %s rows %d bar %s title "%s"', r.out_dir, mat2str(cd(:)'), ...
    mat2str(cl), size(cm, 1), strjoin(cbt(:)', ' '), ttl), n_fail);
% 12c. the same under 'zero': 6 levels 0..5 -> CLim [-0.5 5.5], parula(6); the
%      bar is cropped at the floor, Limits [2.5 5.5], 3 4 5+; folder gains _ramp0.
c.RAMP_FROM = 'zero';
r = plot_marker_slices(fn, c);
[~, ~, cd, ~, ~, cl, cm, cbl, cbt, ~, cbk] = read_points(fullfile(r.out_dir, 'slice_001.fig'));
want = fullfile(root, 'preprocessing', 'mScarlet_plots_dtc', 'qc0_0', 'full', 'ge3_sat5_ramp0');
ok = strcmp(r.out_dir, want) && isequal(cd(:)', [3 5]) && ...
    max(abs(cl - [-0.5 5.5])) < 1e-12 && isequal(size(cm), [6 3]) && ...
    max(abs(cm - parula(6)), [], 'all') < 1e-12 && ...
    max(abs(cbl - [2.5 5.5])) < 1e-12 && isequal(cbt(:)', {'3', '4', '5+'}) && ...
    isequal(cbk(:)', [3 4 5]);
n_fail = report(ok, '12c. floor 3, ceiling 5, ramp from zero: ge3_sat5_ramp0, CLim [-0.5 5.5], parula(6), bar cropped 3..5+', ...
    sprintf('%s CData %s CLim %s rows %d bar %s', r.out_dir, mat2str(cd(:)'), mat2str(cl), ...
    size(cm, 1), strjoin(cbt(:)', ' ')), n_fail);
% 12d. errors.
c = base; c.DRY_RUN = true; c.MIN_ROLONIES = 5; c.ROLONY_CEILING = 5;
msg = error_message(@() plot_marker_slices(fn, c));
n_fail = report(contains(msg, 'below the ceiling'), ...
    '12d-1. floor 5 = ceiling 5 under ''floor'' errors', sprintf('message: %s', msg), n_fail);
c = base; c.DRY_RUN = true; c.MIN_ROLONIES = 2.5;
msg = error_message(@() plot_marker_slices(fn, c));
n_fail = report(contains(msg, 'MIN_ROLONIES') && contains(msg, 'whole number'), ...
    '12d-2. floor 2.5 errors', sprintf('message: %s', msg), n_fail);
c = base; c.DRY_RUN = true; c.ROLONY_CEILING = 7.5;
msg = error_message(@() plot_marker_slices(fn, c));
n_fail = report(contains(msg, 'ROLONY_CEILING') && contains(msg, 'whole number'), ...
    '12d-3. ceiling 7.5 errors', sprintf('message: %s', msg), n_fail);
c = base; c.DRY_RUN = true; c.RAMP_FROM = 'bogus';
msg = error_message(@() plot_marker_slices(fn, c));
n_fail = report(contains(msg, 'RAMP_FROM'), ...
    '12d-4. RAMP_FROM = ''bogus'' errors', sprintf('message: %s', msg), n_fail);
% 12e. under 'zero' floor 5 = ceiling 5 only warns: slice 1 draws c3 (20) as 5,
%      bar Limits [4.5 5.5] labelled 5+.
c = base; c.READS_THRESH = 0; c.GENES_THRESH = 0; c.MIN_ROLONIES = 5;
c.ROLONY_CEILING = 5; c.RAMP_FROM = 'zero';
% The warning is left on so lastwarn records it; it prints once here, expected.
lastwarn('');
msg = error_message(@() plot_marker_slices(fn, c));
[~, warn_id] = lastwarn;
out = fullfile(root, 'preprocessing', 'mScarlet_plots_dtc', 'qc0_0', 'full', 'ge5_sat5_ramp0');
ok = isempty(msg) && strcmp(warn_id, 'plot_marker_slices:saturated') && ...
    isfile(fullfile(out, 'slice_001.fig'));
if ok
    [~, ~, cd, ~, ~, ~, ~, cbl, cbt, ~, cbk] = read_points(fullfile(out, 'slice_001.fig'));
    ok = isequal(cd, 5) && max(abs(cbl - [4.5 5.5])) < 1e-12 && ...
        isequal(cbt(:)', {'5+'}) && isequal(cbk, 5);
end
n_fail = report(ok, '12e. ramp from zero, floor 5 = ceiling 5: warns (saturated), draws c3 as 5, bar 5+', ...
    sprintf('error "%s", last warning id "%s", folder %s', msg, warn_id, out), n_fail);
% 12f. one ceiling serves both markers: GCaMP at ceiling 5 is ge0_sat5.
c = base; c.DRY_RUN = true; c.MARKER = 'gcamp'; c.READS_THRESH = 0; c.GENES_THRESH = 0;
c.ROLONY_CEILING = 5;
r = plot_marker_slices(fn, c);
want = fullfile(root, 'preprocessing', 'GCaMP_plots_dtc', 'qc0_0', 'full', 'ge0_sat5');
n_fail = report(strcmp(r.out_dir, want), '12f. GCaMP with ROLONY_CEILING = 5 writes ge0_sat5', ...
    sprintf('got %s', r.out_dir), n_fail);
% 12g. 'zero' at the default ceiling reproduces the previous behaviour: mScarlet
%      floor 3 -> ge3_sat15_ramp0, 16 levels 0..15, CLim [-0.5 15.5], parula(16),
%      CData [3 15]; bar cropped to [2.5 15.5], step ceil(13/10) = 2, ticks
%      3 5 7 9 11 13 15 labelled 3 5 7 9 11 13 15+.
c = base; c.READS_THRESH = 0; c.GENES_THRESH = 0; c.MIN_ROLONIES = 3; c.RAMP_FROM = 'zero';
r = plot_marker_slices(fn, c);
[~, ~, cd, ~, ~, cl, cm, cbl, cbt, ~, cbk] = read_points(fullfile(r.out_dir, 'slice_001.fig'));
want = fullfile(root, 'preprocessing', 'mScarlet_plots_dtc', 'qc0_0', 'full', 'ge3_sat15_ramp0');
ok = strcmp(r.out_dir, want) && isequal(cd(:)', [3 15]) && ...
    max(abs(cl - [-0.5 15.5])) < 1e-12 && isequal(size(cm), [16 3]) && ...
    max(abs(cm - parula(16)), [], 'all') < 1e-12 && ...
    max(abs(cbl - [2.5 15.5])) < 1e-12 && isequal(cbk(:)', [3 5 7 9 11 13 15]) && ...
    isequal(cbt(:)', {'3', '5', '7', '9', '11', '13', '15+'});
n_fail = report(ok, '12g. ramp from zero, default ceiling: ge3_sat15_ramp0, CLim [-0.5 15.5], parula(16), bar 3..15+', ...
    sprintf('%s CData %s CLim %s rows %d ticks %s', r.out_dir, mat2str(cd(:)'), mat2str(cl), ...
    size(cm, 1), mat2str(cbk)), n_fail);
% 12h. the resume key holds resolved values: the folder of runs{1} (mScarlet
%      20/5 ge0, blank ceiling, 'floor', last written by 9e) matches the same
%      run with the ceiling typed as 15 and RAMP_FROM spelled 'Floor'.
c = base; c.SKIP_EXISTING = true; c.ROLONY_CEILING = 15; c.RAMP_FROM = 'Floor';
r = plot_marker_slices(fn, c);
n_fail = report(r.skipped && strcmp(r.out_dir, runs{1}.out_dir), ...
    '12h. blank ceiling = typed default, and ''Floor'' = ''floor'', for resume: skipped', ...
    sprintf('skipped = %d, %s', r.skipped, r.out_dir), n_fail);

% -- Real data (optional) --------------------------------------------------------
if isempty(real_filt_neurons)
    fprintf('  SKIP  real-data run -- pass filt_neurons to include it\n');
else
    n_fail = check_real(real_filt_neurons, base, fullfile(root, 'real'), n_fail);
end

fprintf('\n%d failure(s)\n', n_fail);
end


function n_fail = check_real(fn, base, root, n_fail)
% mScarlet, 20/5, cutoff 1, recomputed here with findgroups/splitapply rather
% than the masks plot_marker_slices builds.
c = base; c.MIN_ROLONIES = 1; c.ANALYSIS_ROOT_OVERRIDE = root; c.PNG_DPI = 72;
t = tic;
r = plot_marker_slices(fn, c);
fprintf('        real run: %u figures in %.0f s\n', height(r.slices), toc(t));

s = double(fn.slice(:));
reads = full(sum(fn.expmat, 2));
genes = full(sum(fn.expmat > 0, 2));
msc = full(fn.expmat(:, 114));
keep = ~isnan(s) & reads >= 20 & genes >= 5;
[g, sl] = findgroups(s(keep));
n_qc = splitapply(@numel, s(keep), g);
n_drawn = splitapply(@(x) nnz(x >= 1), msc(keep), g);
med = splitapply(@median, reads(keep), g);

T = readtable(fullfile(r.out_dir, 'median_total_reads.csv'));
ok = isequal(T.slice(:), sl(:)) && isequal(T.cells_passing_qc(:), n_qc(:)) && ...
    isequal(T.cells_drawn(:), n_drawn(:)) && max(abs(T.median_total_reads(:) - med(:))) < 1e-9;
n_fail = report(ok, 'R1. real CSV matches an independent count per slice', ...
    sprintf('%u CSV rows vs %u slices recomputed', height(T), numel(sl)), n_fail);

has = ~isnan(s);
[ga, sa] = findgroups(s(has));
p = double(fn.pos(has, :)) * r.um_per_pos;
cx = (splitapply(@min, p(:, 1), ga) + splitapply(@max, p(:, 1), ga)) / 2;
cy = (splitapply(@min, p(:, 2), ga) + splitapply(@max, p(:, 2), ga)) / 2;
extent = @(v) max(v) - min(v);   % not range(): that needs the Statistics Toolbox
w = max([splitapply(extent, p(:, 1), ga); splitapply(extent, p(:, 2), ga)]);
[~, loc] = ismember(r.slices.slice, sa);
ok = abs(r.span_um - w) < 1e-6 && ...
    max(abs(r.slices.xlim - (cx(loc) + [-w w] / 2)), [], 'all') < 1e-6 && ...
    max(abs(r.slices.ylim - (cy(loc) + [-w w] / 2)), [], 'all') < 1e-6 && ...
    all(r.slices.n_outside == 0);
n_fail = report(ok, 'R2. real windows match an independent all-cell bounding box; nothing clipped', ...
    sprintf('span %g vs %g um', r.span_um, w), n_fail);
end


function row = make_row(reads, genes, msc, gcamp)
% An expmat row with exactly `reads` total counts over exactly `genes`
% non-zero columns, mScarlet in 114 and GCaMP in 112. The other detected genes
% take columns 1.. -- one count each, column 1 the remainder.
row = zeros(1, 114);
row(114) = msc;
row(112) = gcamp;
others = genes - (msc > 0) - (gcamp > 0);
rest = reads - msc - gcamp;
assert(others >= 0 && rest >= others && (others > 0 || rest == 0), ...
    'make_row(%d, %d, %d, %d) cannot be built', reads, genes, msc, gcamp);
if others > 0
    row(1:others) = 1;
    row(1) = row(1) + rest - others;
end
end


function [xd, yd, cd, xl, yl, cl, cm, cbl, cbt, ttl, cbk] = read_points(fig_path)
% From a saved figure: the coloured scatter's points, the axes limits, colour
% limits and colormap, the colorbar's limits, tick labels and tick positions,
% and the title.
% DRAW_BELOW_CUTOFF is off throughout, so there is at most one scatter object.
g = openfig(fig_path, 'invisible');
h = findobj(g, 'Type', 'scatter');
ax = findobj(g, 'Type', 'axes');
cb = findobj(g, 'Type', 'colorbar');
xl = ax(1).XLim;
yl = ax(1).YLim;
cl = ax(1).CLim;
cm = ax(1).Colormap;
cbl = cb(1).Limits;
cbt = cellstr(cb(1).TickLabels);
cbk = cb(1).Ticks;
ttl = char(get(get(ax(1), 'Title'), 'String'));
if isempty(h)
    xd = []; yd = []; cd = [];
else
    xd = h(1).XData; yd = h(1).YData; cd = h(1).CData;
end
close(g);
end


function msg = error_message(fcn)
msg = '';
try
    fcn();
catch err
    msg = err.message;
end
end


function n_fail = report(ok, what, detail, n_fail)
if ok
    fprintf('  PASS  %s\n', what);
else
    fprintf('  FAIL  %s\n        %s\n', what, detail);
    n_fail = n_fail + 1;
end
end
