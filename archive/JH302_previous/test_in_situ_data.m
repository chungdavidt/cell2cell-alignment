clear all
close all

load('./filt_neurons.mat');

reads_thresh=0;
genes_thresh=0;

uniq_slices=unique(filt_neurons.slice);
uniq_slices = uniq_slices(~isnan(uniq_slices))
countspercell=full(filt_neurons.expmat(:,114));
total_cells = numel(countspercell);
pass_qc=sum(filt_neurons.expmat,2)>=reads_thresh&sum(filt_neurons.expmat>0,2)>=genes_thresh;
total_passed = nnz(pass_qc);
Percent_passed = total_passed/total_cells*100
median(sum(filt_neurons.expmat(pass_qc,:),2))
counts = [];

% plot 1st slice
figure; hold on
for nn=cell2mat(filt_neurons.uniq_slice{1})'
    in_slice=filt_neurons.slice==nn;
    plot(filt_neurons.pos(in_slice&pass_qc,1),-filt_neurons.pos(in_slice&pass_qc,2), '.r');
    %
end
title(['filt neurons pos 1']);


%% align filt_neuron to stiched images - see conversion in notes
for i_slice = 1:length(filt_neurons.uniq_slice)
    im_stitch = imread(['.\checkregistration\Pos',num2str(i_slice),'RGBalignedn2vhyb01.tif']);

    figure
    image(im_stitch); hold on

    % figure; hold on  - plot filt-neuron on stitched image
    for nn=cell2mat(filt_neurons.uniq_slice{i_slice})'
        in_slice=filt_neurons.slice==nn;
        plot(filt_neurons.pos(in_slice&pass_qc,1)/5,filt_neurons.pos(in_slice&pass_qc,2)/5, '.r');
        %
    end
    title(['stitched POS',num2str(i_slice)]);
    keyboard
end


% 
% for nn= 1:numel(uniq_slices)
%     in_slice=filt_neurons.slice==uniq_slices(nn);
%     figure('Position',[50 50 600 400]);
%     med_count= full(median(sum(filt_neurons.expmat(in_slice&pass_qc,:),2)));
%     disp('Slice number: ' + string(nn) + '   Median count: ' + string(med_count));
%     scatter(filt_neurons.pos(in_slice&pass_qc,1),filt_neurons.pos(in_slice&pass_qc,2), ...
%         5,...
%         countspercell(in_slice&pass_qc),...
%         'filled');
%     clim([0 10]);
%     set(gca,'ydir','reverse')
%     colorbar;
%     title(sprintf('slice %u',uniq_slices(nn)));
%     disp('_______________________________________________________________')
%     
%     
%     keyboard
%     
%     counts = [counts; med_count];
%     if ~exist('mScarlet_plots', 'dir')
%         mkdir('mScarlet_plots')
%     end
%     cd('mScarlet_plots')
%     savefig('Slice number' + string(nn))
%     cd ..
%     
% end


%% align filt_neuron to HYB (raw cell mask info) - get one section
for i_slice = [1 5]
    
    % determine # of FOV in x and y from image size of stitched image ->
    % see notes on the conversion
    im_stitch = imread(['.\checkregistration\Pos',num2str(i_slice),'RGBalignedn2vhyb01.tif']);
    num_of_FOV_x = round((max(size(im_stitch,2)*10+710)-735)/2465);
    num_of_FOV_y = round((max(size(im_stitch,1)*10+710)-735)/2465);

    for i_x = 0:(num_of_FOV_x-1)
        for i_y = 0:(num_of_FOV_y-1)

            x_str = num2str(i_x);
            while length(x_str)<3
                x_str = ['0',x_str];
            end

            y_str = num2str(i_y);
            while length(y_str)<3
                y_str = ['0',y_str];
            end

            hyb_file = ['.\hyb\MAX_Pos',num2str(i_slice),'_',x_str,'_',y_str,'\alignedn2vhyb01.tif'];
            im_hyb = imread(hyb_file);

            istr1 = findstr(hyb_file,'_');
            istr2 = findstr(hyb_file,'\');
            FOV_x_pos = str2num(hyb_file(istr1(2)+1:istr1(3)-1));
            FOV_y_pos = str2num(hyb_file(istr1(3)+1:istr2(3)-1));


            figure
            image(im_hyb)

            x_offset = -(2465*(num_of_FOV_x-1-FOV_x_pos)); %9860
            y_offset = -(2465*FOV_y_pos); %24660

            hold on
            for nn=cell2mat(filt_neurons.uniq_slice{i_slice})'
                in_slice=filt_neurons.slice==nn;
                plot(filt_neurons.pos(in_slice&pass_qc,1)*2+x_offset,filt_neurons.pos(in_slice&pass_qc,2)*2+y_offset, '.r');
                %
            end

            keyboard

        end
    end

end