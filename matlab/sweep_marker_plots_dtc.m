% sweep_marker_plots_dtc.m -- the marker scatter for every combination of
% rolony ceiling, marker, QC pair and rolony cutoff.
%
% Expects filt_neurons in the workspace. Each combination is one call to
% plot_marker_slices.m and writes its own folder of the usual hierarchy,
%   <Marker>_plots_dtc\qc<reads>_<genes>\full\ge<cut>_sat<ceiling>\
% A slice's window comes from all of its cells before QC, so its figures share
% one frame across every combination and can be flipped through directly.
% Colours run from the floor to the ceiling (RAMP_FROM = 'floor'), so compare
% colours only between figures with the same ge<cut>_sat<ceiling>. The ceiling
% is the outermost loop: every run at the first ceiling finishes before the next
% starts.
%
% SKIP_EXISTING is on: a folder whose plot_settings.txt matches these settings
% is skipped, so re-running after an interruption picks up where it stopped.
% DRY_RUN starts true -- it lists the output folders and writes nothing. Check
% the list, then set it false.

%% ---- SWEEP ----------------------------------------------------------------
CEILINGS = [5 10 15];   % one ceiling for both markers
MARKERS  = {'mscarlet', 'gcamp'};
READS    = 20:-5:0;
GENES    = 5:-1:0;
ROLONIES = 0:3;
EXPECTED_RUNS = 720;   % numel of the five lists multiplied; update with them

DRY_RUN  = true;

%% ---- FIXED SETTINGS (what each one does: gen_marker_plots_dtc.m) ----------
cfg = struct();   % reset: a cfg left in the workspace by another script is not reused
cfg.DRAW_BELOW_CUTOFF             = false;
cfg.BELOW_COLOR                   = [0.25 0.25 0.25];
cfg.COLORMAP                      = 'parula';
cfg.CROP_TO_SUBSLICE              = false;
cfg.SUBSLICE_DEFINITIONS_OVERRIDE = '';
cfg.PANEL_COLUMNS                 = 114;
cfg.FIG_SIZE                      = [600 600];
cfg.MARKER_SIZE                   = 5;
cfg.PNG_DPI                       = 300;
cfg.AXIS_SPAN_UM                  = [];
cfg.RAMP_FROM                     = 'floor';
cfg.ANALYSIS_ROOT_OVERRIDE        = '';
cfg.SKIP_EXISTING                 = true;
cfg.VERBOSE                       = false;
cfg.DRY_RUN                       = DRY_RUN;
%% ---------------------------------------------------------------------------

assert(exist('filt_neurons', 'var') == 1, ...
    'filt_neurons is not in the workspace -- load the brain''s filt_neurons.mat first.');

n_runs = numel(CEILINGS) * numel(MARKERS) * numel(READS) * numel(GENES) * numel(ROLONIES);
assert(n_runs == EXPECTED_RUNS, ...
    ['The lists give %u runs, EXPECTED_RUNS says %u. Check CEILINGS, MARKERS, ' ...
     'READS, GENES and ROLONIES, then update EXPECTED_RUNS.'], n_runs, EXPECTED_RUNS);

fprintf('%u runs, DRY_RUN = %d\n', n_runs, DRY_RUN);
t_sweep = tic;
k = 0;
n_written = 0;
n_skipped = 0;
% (:)' so a list typed as a column still iterates element by element.
for ceiling = CEILINGS(:)'
    for marker = MARKERS(:)'
        for reads = READS(:)'
            for genes = GENES(:)'
                for rolonies = ROLONIES(:)'
                    k = k + 1;
                    run_cfg = cfg;
                    run_cfg.ROLONY_CEILING = ceiling;
                    run_cfg.MARKER         = marker{1};
                    run_cfg.READS_THRESH   = reads;
                    run_cfg.GENES_THRESH   = genes;
                    run_cfg.MIN_ROLONIES   = rolonies;

                    t_run = tic;
                    res = plot_marker_slices(filt_neurons, run_cfg);
                    if res.dry_run
                        status = 'dry run';
                    elseif res.skipped
                        status = 'skipped';
                        n_skipped = n_skipped + 1;
                    else
                        status = 'written';
                        n_written = n_written + 1;
                    end
                    elapsed = char(duration(0, 0, toc(t_sweep), 'Format', 'hh:mm:ss'));
                    fprintf('[%3u/%u] sat%-3g %-8s qc%g_%g ge%g  %-7s %6.1f s  (elapsed %s)  %s\n', ...
                        k, n_runs, ceiling, marker{1}, reads, genes, rolonies, status, ...
                        toc(t_run), elapsed, res.out_dir);
                end
            end
        end
    end
end

fprintf('\nDone: %u written, %u skipped, %u runs, %s\n', n_written, n_skipped, n_runs, ...
    char(duration(0, 0, toc(t_sweep), 'Format', 'hh:mm:ss')));
if DRY_RUN
    fprintf('Dry run -- nothing written. Set DRY_RUN = false to plot.\n');
end
