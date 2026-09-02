% Gen_mScarlet_plots_dtc.m -- Gen_mScarlet_plots.m drawn square: x and y span the same
% range and one unit is the same length on both axes, so the scatter has
% the section's proportions. Saves a .png beside each .fig. Everything
% else is unchanged.

reads_thresh=20;
genes_thresh=5;

uniq_slices=unique(filt_neurons.slice);
uniq_slices = uniq_slices(~isnan(uniq_slices))
countspercell=full(filt_neurons.expmat(:,114));
total_cells = numel(countspercell);
pass_qc=sum(filt_neurons.expmat,2)>=reads_thresh&sum(filt_neurons.expmat>0,2)>=genes_thresh;
total_passed = nnz(pass_qc);
Percent_passed = total_passed/total_cells*100
median(sum(filt_neurons.expmat(pass_qc,:),2))
counts = [];

for nn= 1:numel(uniq_slices)
    in_slice=filt_neurons.slice==uniq_slices(nn);
    f = figure('Position',[50 50 600 600]);
    med_count= full(median(sum(filt_neurons.expmat(in_slice&pass_qc,:),2)));
    disp('Slice number: ' + string(nn) + '   Median count: ' + string(med_count));
    scatter(filt_neurons.pos(in_slice&pass_qc,1),filt_neurons.pos(in_slice&pass_qc,2), ...
        5,...
        countspercell(in_slice&pass_qc),...
        'filled');
    clim([0 5]);
    set(gca,'ydir','reverse')
    axis image
    lims = [min([xlim ylim]) max([xlim ylim])];
    xlim(lims); ylim(lims);
    colorbar;
    title(sprintf('slice %u',uniq_slices(nn)));
    disp('_______________________________________________________________')
    counts = [counts; med_count];
    if ~exist('mScarlet_plots_dtc', 'dir')
        mkdir('mScarlet_plots_dtc')
    end
    cd('mScarlet_plots_dtc')
    savefig('Slice number' + string(nn))
    exportgraphics(f, 'Slice number' + string(nn) + '.png', 'Resolution', 300)
    cd ..
    close(f);
end
