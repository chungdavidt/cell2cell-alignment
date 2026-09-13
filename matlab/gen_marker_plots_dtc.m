% gen_marker_plots_dtc.m -- per-slice marker scatter, one marker and one set of
% thresholds per run.
%
% Replaces Gen_mScarlet_plots_dtc.m and Gen_GCaMP_plots_dtc.m: the marker is a
% config value below, not a separate file. Expects filt_neurons already in the
% workspace, as the originals do.
%
% This file is the settings. The plotting is plot_marker_slices.m, whose header
% documents the colour rule, the frame, the output layout and SKIP_EXISTING.
% sweep_marker_plots_dtc.m runs every combination of marker, QC pair and cutoff.

%% ---- CONFIG ---------------------------------------------------------------
cfg = struct();   % reset: a cfg left in the workspace by another script is not reused

% -- Marker ----------------------------------------------------------------
cfg.MARKER        = 'mscarlet';   % 'mscarlet' | 'gcamp'

% -- QC --------------------------------------------------------------------
% QC floors for a cell to be plotted at all. Deliberately NOT inherited from
% local_config.py's QC_MIN_READS / QC_MIN_GENES -- those gate the alignment
% pipeline, these gate a figure. Per-brain either way: read the dataset's own
% Gen_*_plots.m rather than carrying another brain's numbers. Note the two
% copies disagree on purpose -- Gen_mScarlet_plots.m in THIS folder is 20/5,
% the lab's own copy is 0/0 so no marker detection is gated on transcriptome
% quality. Two questions, two answers; feedback_qc_thresholds_are_per_dataset.md.
% The subslice crop was chosen at the config's pair, not these -- see
% CROP_TO_SUBSLICE below.
cfg.READS_THRESH  = 0;
cfg.GENES_THRESH  = 0;

% -- Rolony cutoff and colour ----------------------------------------------
% Rolony cutoff: a cell below this is not painted. 0 draws every QC-passing
% cell, as Gen_*_plots.m did; 1 draws every marker+ cell. The marker's step-4
% draw floor is 5 (mScarlet) / 3 (GCaMP) if you want this figure to match what
% the pipeline renders. Changing it does NOT change any remaining cell's colour.
cfg.MIN_ROLONIES  = 0;   % must be >= 0

% Cells below MIN_ROLONIES as a flat grey, for when the section outline is
% wanted behind a cutoff above 0. At a cutoff of 0 nothing is below it.
% 0.25 is the grey the pipeline's cellmask field paints at.
cfg.DRAW_BELOW_CUTOFF = false;
cfg.BELOW_COLOR       = [0.25 0.25 0.25];

% The span of the count -> colour mapping is NOT here on purpose: it is
% RAMP_MAX in the marker table in plot_marker_slices.m, a fixed per-marker
% constant. Its slope is (top colour - bottom colour) / RAMP_MAX, so a per-run
% dial would change the slope between runs -- exactly what the absolute ramp
% exists to prevent.

% Palette. Any MATLAB colormap function name -- 'parula' (blue -> yellow),
% 'turbo', 'hot', 'jet' -- or 'marker' for this marker's dark-to-bright anchors
% from marker_profiles.py. Either way it is sampled into RAMP_MAX + 1 discrete
% levels, one per count 0..RAMP_MAX. 'marker' does NOT reproduce the cellmask
% renders' colours: those span [1, ceiling] and leave count 0 unpainted.
cfg.COLORMAP      = 'parula';

% -- Crop ------------------------------------------------------------------
% Keep only cells in the FOVs identify_marker_subslices.py picked for each
% slice -- the largest 8-connected cluster of FOVs holding a QC-passing marker+
% cell, plus bridges. Crops the far-field scatter; it will not hug the labelled
% region, because the unit is a whole ~1 mm FOV tile.
% This is a separate knob from MIN_ROLONIES: the cutoff drops low-count cells
% everywhere, the crop drops cells by location whatever their count.
% true writes under crop\, false under full\. The window does not move with
% the crop: it is framed on every cell of the slice either way.
%
% The FOVs were picked at local_config.py's QC_MIN_READS / QC_MIN_GENES, NOT at
% READS_THRESH / GENES_THRESH above. With the two pairs unequal, the plotted
% cells follow READS_THRESH / GENES_THRESH but the crop region is still the one
% chosen at the config's pair, and the qc folder name records only this
% script's pair. To move the crop with the QC, change the config and re-run
% identify_marker_subslices.py --marker <marker> before plotting.
cfg.CROP_TO_SUBSLICE = false;

% Blank -> <ANALYSIS_ROOT>\preprocessing\subslice_definitions\ and the file
% matching MARKER: subslice_definitions.mat for mScarlet (the one the pipeline
% reads), subslice_definitions_<marker>.mat for anything else.
cfg.SUBSLICE_DEFINITIONS_OVERRIDE = '';

% -- Guards ----------------------------------------------------------------
% Column count of this brain's panel. Guard only, and it asserts equality: a
% panel with a different slot order but enough columns would otherwise plot the
% wrong gene with no error.
cfg.PANEL_COLUMNS = 114;

% -- Figure ----------------------------------------------------------------
cfg.FIG_SIZE      = [600 600];    % pixels, [width height]
cfg.MARKER_SIZE   = 5;            % scatter point area
cfg.PNG_DPI       = 300;

% Window width and height in µm, the same for every slice; each slice is
% centred on the bounding box of all its cells. Blank -> the widest slice,
% measured on every cell before QC, cutoff or crop, so it is one number for
% every run on this brain. Set a number to override, e.g. to draw two brains
% at one scale; cells past the edge of a smaller window are not visible, and
% the run warns per slice.
cfg.AXIS_SPAN_UM  = [];

% -- Output ----------------------------------------------------------------
% Blank -> read ANALYSIS_ROOT from local_config.py one level up. Set a path
% here to write somewhere else instead.
cfg.ANALYSIS_ROOT_OVERRIDE = '';

% -- Run control -----------------------------------------------------------
cfg.SKIP_EXISTING = false;   % true: skip if the folder's plot_settings.txt matches these settings
cfg.VERBOSE       = true;    % false: print nothing but warnings
cfg.DRY_RUN       = false;   % true: print the output folder and window, write nothing
%% ---------------------------------------------------------------------------

assert(exist('filt_neurons', 'var') == 1, ...
    'filt_neurons is not in the workspace -- load the brain''s filt_neurons.mat first.');

result = plot_marker_slices(filt_neurons, cfg);
if result.dry_run
    fprintf('dry run: would write %s\n  window %g um, %.4g um per pos unit\n', ...
        result.out_dir, result.span_um, result.um_per_pos);
end
