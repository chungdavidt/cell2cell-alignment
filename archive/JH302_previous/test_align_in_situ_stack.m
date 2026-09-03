clear all
close all

% determine # of FOV in x and y from image size of stitched image ->
% see notes on the conversion
i_slice = 1;

im_stitch = imread(['.\checkregistration\Pos',num2str(i_slice),'RGBalignedn2vhyb01.tif']);

num_of_FOV_x = round((max(size(im_stitch,2)*10+710)-735)/2465);
num_of_FOV_y = round((max(size(im_stitch,1)*10+710)-735)/2465);


load('./filt_neurons.mat');

reads_thresh=0;
genes_thresh=0;

uniq_slices=unique(filt_neurons.slice);
uniq_slices = uniq_slices(~isnan(uniq_slices));
countspercell=full(filt_neurons.expmat(:,114));
total_cells = numel(countspercell);
pass_qc=sum(filt_neurons.expmat,2)>=reads_thresh&sum(filt_neurons.expmat>0,2)>=genes_thresh;
total_passed = nnz(pass_qc);
Percent_passed = total_passed/total_cells*100;
median(sum(filt_neurons.expmat(pass_qc,:),2));
counts = [];

% plot each slice
raw_image_stack = {}; n_raw_im = 0;
for nn=cell2mat(filt_neurons.uniq_slice{i_slice})'
    
    n_raw_im = n_raw_im+1;
    % 
    in_slice=filt_neurons.slice==nn;
    % figure; hold on
    % plot(filt_neurons.pos(in_slice&pass_qc,1),-filt_neurons.pos(in_slice&pass_qc,2), '.r');


    % find the corresponding HYB
    fov_names = unique(filt_neurons.fov_names(filt_neurons.fov(in_slice&pass_qc)));

    im_tmp = uint16(1);

    for i_fov = 1:length(fov_names)

        %hyb_file = ['.\hyb\MAX_Pos',num2str(i_slice),'_',x_str,'_',y_str,'\alignedn2vhyb01.tif'];
        hyb_file = ['.\hyb\',fov_names{i_fov},'\alignedn2vhyb01.tif'];
        im_hyb = imread(hyb_file);

        load(['.\hyb\',fov_names{i_fov},'\cellmask.mat']);
        cellmask = maski;

        istr1 = findstr(hyb_file,'_');
        istr2 = findstr(hyb_file,'\');
        FOV_x_pos = str2num(hyb_file(istr1(2)+1:istr1(3)-1));
        FOV_y_pos = str2num(hyb_file(istr1(3)+1:istr2(3)-1));

        %figure
        %image(im_hyb)

        x_offset = (2465*(num_of_FOV_x-1-FOV_x_pos)); %9860
        y_offset = (2465*FOV_y_pos); %24660

        %im_tmp(y_offset+1:y_offset+3200, x_offset+1:x_offset+3200) = im_hyb;%(166:3200-166, 204:3200-143);

        im_tmp(y_offset+1+165:y_offset+3200-166, x_offset+1+203:x_offset+3200-143) = im_hyb(166:3200-166, 204:3200-143);

    end

    % figure
    % image(im_tmp); hold on
    % plot(filt_neurons.pos(in_slice&pass_qc,1)*2,filt_neurons.pos(in_slice&pass_qc,2)*2, '.r');

    raw_image_stack{n_raw_im,1} = im_tmp;
    raw_image_stack{n_raw_im,2} = [filt_neurons.pos(in_slice&pass_qc,1)*2 filt_neurons.pos(in_slice&pass_qc,2)*2];

end
title(['filt neurons pos 1']);


for i_im = 2:length(raw_image_stack)
    im1 = raw_image_stack{i_im-1,1};      % fixed
    im2 = raw_image_stack{i_im,1};        % moving

    % trim size
    x_offset = min(raw_image_stack{i_im-1,2}(:,1));
    y_offset = min(raw_image_stack{i_im-1,2}(:,2));
    im1 = im1((1:12000)+y_offset, (1:9000)+x_offset);
    x_offset = min(raw_image_stack{i_im,2}(:,1));
    y_offset = min(raw_image_stack{i_im,2}(:,2));
    im2 = im2((1:12000)+y_offset, (1:9000)+x_offset);

    %imshowpair(im1,im2,"Scaling","joint")
    figure
    subplot(1,2,1); image(im1);
    subplot(1,2,2); image(im2);
    keyboard

%     im1 = imresize(im1, .01);
%     im2 = imresize(im2, .01);
%     keyboard

    tic
    [optimizer,metric] = imregconfig("multimodal");
    toc
    movingRegistered = imregister(im2,im1,"affine",optimizer,metric);
    toc
    
    %imshowpair(im1,movingRegistered,"Scaling","joint")
    figure
    subplot(1,2,1); image(im1);
    subplot(1,2,2); image(im2);
    keyboard

end




















%% ================= junk ==================

    keyboard

    %
    %     % find the corresponding HYB
    %     x_min = min(filt_neurons.pos(in_slice&pass_qc,1)*2);
    %     x_max = max(filt_neurons.pos(in_slice&pass_qc,1)*2);
    %
    %     x_max_FOV_pos = num_of_FOV_x-1-floor(x_min/2465);
    %     x_min_FOV_pos = num_of_FOV_x-1-floor(x_max/2465);
    %     x_max_FOV_pos = min([num_of_FOV_x x_max_FOV_pos]); % truncate at total X FOV - note the inversion due to starting from FOV counting left side
    %     x_min_FOV_pos = max([0 x_min_FOV_pos]); % truncate at 0
    %
    %     y_min = min(filt_neurons.pos(in_slice&pass_qc,2));
    %     y_max = max(filt_neurons.pos(in_slice&pass_qc,2));
    %
    %     y_min_FOV_pos = floor(x_min/2465);
    %     y_max_FOV_pos = ceil(x_max/2465);
    %     y_max_FOV_pos = min([num_of_FOV_y y_max_FOV_pos]);
    %
    %
    %     % load HYB file
    %     figure
    %     n_FOV_x = x_max_FOV_pos-x_min_FOV_pos;
    %     n_FOV_y = y_max_FOV_pos-y_min_FOV_pos;
    %     %im_tmp = uint16(zeros(n_FOV_x*2465+735, n_FOV_y*2465+735));
    %     im_tmp = uint16(zeros(n_FOV_x*2869+735, n_FOV_y*2854));
    %     n_x = 0;
    %     for i_x = x_min_FOV_pos:x_max_FOV_pos
    %         n_x = n_x+1;
    %         n_y = 0;
    %         for i_y = y_min_FOV_pos:y_max_FOV_pos
    %             n_y = n_y+1;
    %
    %             x_str = num2str(i_x);
    %             while length(x_str)<3
    %                 x_str = ['0',x_str];
    %             end
    %
    %             y_str = num2str(i_y);
    %             while length(y_str)<3
    %                 y_str = ['0',y_str];
    %             end
    %
    %             hyb_file = ['.\hyb\MAX_Pos',num2str(i_slice),'_',x_str,'_',y_str,'\alignedn2vhyb01.tif'];
    %             im_hyb = imread(hyb_file);
    %
    %             %im_tmp((n_y-1)*2465+1:(n_y-1)*2465+3200, (n_FOV_x-n_x+1)*2465+1:(n_FOV_x-n_x+1)*2465+3200) = im_hyb;
    %
    %             %im_tmp((n_y-1)*2869+1:(n_y-1)*2869+2869, (n_FOV_x-n_x+1)*2854+1:(n_FOV_x-n_x+1)*2854+2854) = im_hyb(332:3200, 347:3200);
    %             im_tmp((n_y-1)*2869+1:(n_y-1)*2869+2869, (n_FOV_x-n_x+1)*2854+1:(n_FOV_x-n_x+1)*2854+2854) = im_hyb(166:3200-166, 204:3200-143);
    %
    %             %2869        2854
    %
    %             %im_tmp = im_tmp();
    %
    %         end
    %     end
    %
    %             image(im_tmp);
    %
    %
    %             keyboard

