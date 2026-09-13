% test_gen_marker_plots_assumptions.m -- check, on this MATLAB, every assumption
% plot_marker_slices.m (called by gen_marker_plots_dtc.m) rests on.
%
% Each of these was reasoned about on the WSL edit host, where there is no
% MATLAB, so none of them was verified before the script was written. Run this
% once on the execution host; anything that reports FAIL is a real defect in
% gen_marker_plots_dtc.m, not in this file.
%
%   test_gen_marker_plots_assumptions                     % everything but 7-8
%   test_gen_marker_plots_assumptions('<path to subslice_definitions.mat>')
%
% Writes nothing outside tempdir and leaves no figures open.

function test_gen_marker_plots_assumptions(defs_path)
if nargin < 1, defs_path = ''; end
fprintf('MATLAB %s\n\n', version);
n_fail = 0;

% -- 1. count k lands in colormap row k+1 -----------------------------------
% The whole discrete-legend design depends on clim([-0.5 K+0.5]) against a
% (K+1)-row colormap mapping integer k in 0..K to row k+1. Measured by rendering
% and reading pixels back, not by re-deriving MATLAB's documented index formula.
K = 10;
n_rows = K + 1;
cmap = [(1:n_rows)'/n_rows, zeros(n_rows, 1), zeros(n_rows, 1)];   % row r = [r/n_rows 0 0]
f = figure('Visible', 'off', 'Position', [10 10 n_rows*20 40]);
ax = axes('Parent', f, 'Position', [0 0 1 1]);
image(ax, 'XData', [0 K], 'YData', [1 1], 'CData', 0:K, 'CDataMapping', 'scaled');
colormap(ax, cmap); clim(ax, [-0.5, K + 0.5]);
axis(ax, 'off'); xlim(ax, [-0.5 K+0.5]); ylim(ax, [0.5 1.5]);
frame = getframe(ax); px = frame.cdata;
mid = round(size(px, 1) / 2);
got = zeros(1, n_rows);
for k = 0:K
    col = round((k + 0.5) / n_rows * size(px, 2));
    col = min(max(col, 1), size(px, 2));
    got(k + 1) = round(double(px(mid, col, 1)) / 255 * n_rows) - 1;
end
close(f);
n_fail = report(isequal(got, 0:K), 'count k in 0..K renders in colormap row k+1', ...
    sprintf('counts read back: %s', mat2str(got)), n_fail);

% -- 1b. a ramp starting above 0: count k in s..c lands in row k-s+1 --------
% RAMP_FROM = 'floor' puts clim at [floor-0.5 ceiling+0.5] against a
% (ceiling-floor+1)-row colormap. Same read-back as check 1, with s = 3, c = 10.
s_b = 3; c_b = 10;
rows_b = c_b - s_b + 1;
cmap_b = [(1:rows_b)'/rows_b, zeros(rows_b, 1), zeros(rows_b, 1)];
f = figure('Visible', 'off', 'Position', [10 10 rows_b*20 40]);
ax = axes('Parent', f, 'Position', [0 0 1 1]);
image(ax, 'XData', [s_b c_b], 'YData', [1 1], 'CData', s_b:c_b, 'CDataMapping', 'scaled');
colormap(ax, cmap_b); clim(ax, [s_b - 0.5, c_b + 0.5]);
axis(ax, 'off'); xlim(ax, [s_b - 0.5, c_b + 0.5]); ylim(ax, [0.5 1.5]);
frame = getframe(ax); px = frame.cdata;
mid = round(size(px, 1) / 2);
got_b = zeros(1, rows_b);
for k = s_b:c_b
    col = round((k - s_b + 0.5) / rows_b * size(px, 2));
    col = min(max(col, 1), size(px, 2));
    got_b(k - s_b + 1) = round(double(px(mid, col, 1)) / 255 * rows_b) + s_b - 1;
end
close(f);
n_fail = report(isequal(got_b, s_b:c_b), ...
    'count k in 3..10 renders in colormap row k-2 with clim [2.5 10.5]', ...
    sprintf('counts read back: %s', mat2str(got_b)), n_fail);

% -- 2. cb.Limits crops without changing the data mapping -------------------
f = figure('Visible', 'off');
ax = axes('Parent', f); colormap(ax, cmap); clim(ax, [-0.5, K + 0.5]);
cb = colorbar(ax); before = clim(ax);
cb.Limits = [3.5, K + 0.5]; cb.Ticks = 4:K;
after = clim(ax);
% Read every property BEFORE close(f) -- the colorbar dies with the figure, and
% report()'s detail argument is evaluated whether the check passed or not.
lims_after = cb.Limits;
ok = isequal(before, after) && isequal(lims_after, [3.5, K + 0.5]);
close(f);
n_fail = report(ok, 'setting cb.Limits leaves axes CLim untouched (no re-shading)', ...
    sprintf('CLim %s -> %s, cb.Limits %s', mat2str(before), mat2str(after), ...
    mat2str(lims_after)), n_fail);

% -- 3. CreateFcn set after creation does not fire, and the .fig opens visible
f = figure('Visible', 'off');
set(f, 'CreateFcn', 'set(gcbo,''Visible'',''on'')');
fired_early = strcmp(get(f, 'Visible'), 'on');
tmp = [tempname '.fig'];
savefig(f, tmp); close(f);
g = openfig(tmp); vis = get(g, 'Visible'); close(g); delete(tmp);
n_fail = report(~fired_early && strcmp(vis, 'on'), ...
    'CreateFcn does not fire on set, and a reopened .fig is visible', ...
    sprintf('fired at set: %d, reopened Visible: %s', fired_early, vis), n_fail);

% -- 4. an empty %s argument substitutes nothing and the template CONTINUES --
% Measured on R2026a 2026-09-12, refuting the reasoning this file was written
% against: fprintf does NOT stop at a specifier whose argument is empty. The
% assertion is inverted from its original form on purpose -- it now pins the
% real behaviour, so a future MATLAB that starts truncating would fail here.
s = evalc('fprintf(''A%sB\n'', repmat('' x'', 1, 0));');
n_fail = report(contains(s, 'B'), ...
    'fprintf keeps going when a %s argument is empty (does NOT truncate)', ...
    sprintf('produced %s', mat2str(s)), n_fail);

% -- 5. row == scalar & column implicitly expands to N-by-N -----------------
row = 1:4; colmask = true(4, 1);
sz = size(row == 1 & colmask);
n_fail = report(isequal(sz, [4 4]), ...
    'a row-shaped mask against a column expands to N-by-N instead of erroring', ...
    sprintf('size = %s', mat2str(sz)), n_fail);

% -- 6. xlim rejects a zero-width span --------------------------------------
f = figure('Visible', 'off'); ax = axes('Parent', f); threw = false;
try, xlim(ax, [5 5]); catch, threw = true; end
close(f);
n_fail = report(threw, 'xlim errors on a zero-width span (pad_span is needed)', ...
    sprintf('threw = %d', threw), n_fail);

% -- 7. scatter tolerates empty data and does not read a 3-row CData as RGB -
f = figure('Visible', 'off'); ax = axes('Parent', f);
ok_empty = true;
try, scatter(ax, zeros(0,1), zeros(0,1), 5, zeros(0,1), 'filled'); ...
catch, ok_empty = false; end
h = scatter(ax, [1;2;3], [1;2;3], 5, [1;2;3], 'filled');
cdata_size = size(h.CData);          % before close(f), same reason as above
three_is_cdata = cdata_size(2) == 1;
close(f);
n_fail = report(ok_empty, 'scatter accepts empty data', ...
    sprintf('ok = %d', ok_empty), n_fail);
n_fail = report(three_is_cdata, ...
    'a 3-element COLUMN CData stays colormapped, not read as one RGB triplet', ...
    sprintf('CData is %s', mat2str(cdata_size)), n_fail);

% -- 8. what the definitions file actually looks like in MATLAB -------------
if isempty(defs_path)
    fprintf('  SKIP  subslice_info shape -- pass the path to subslice_definitions.mat\n');
elseif ~isfile(defs_path)
    fprintf('  SKIP  subslice_info shape -- not found: %s\n', defs_path);
else
    d = load(defs_path); si = d.subslice_info;
    n_fail = report(iscell(si), ...
        'subslice_info is a CELL (so entries need {i}, not (i))', ...
        sprintf('class = %s, size = %s', class(si), mat2str(size(si))), n_fail);
    if iscell(si), e = si{1}; else, e = si(1); end
    fl = e.fov_list;
    n_fail = report(ischar(fl), ...
        'fov_list is a CHAR MATRIX (so (:) would scramble it)', ...
        sprintf('class = %s, size = %s', class(fl), mat2str(size(fl))), n_fail);
    if ischar(fl)
        names = cellstr(fl);
        scrambled = fl(:)';
        fprintf('        fov_list(:) would give: %s...\n', scrambled(1:min(30, end)));
        fprintf('        cellstr gives %d name(s), first: %s\n', numel(names), names{1});
        pad = setdiff(double(fl(1, :)), double(strtrim(fl(1, :))));
        if ~isempty(pad)
            fprintf('        padding code point(s): %s (32 = space, 0 = NUL)\n', mat2str(pad));
        end
    end
end

fprintf('\n%d failure(s)\n', n_fail);
end

function n_fail = report(ok, what, detail, n_fail)
if ok
    fprintf('  PASS  %s\n', what);
else
    fprintf('  FAIL  %s\n        %s\n', what, detail);
    n_fail = n_fail + 1;
end
end
