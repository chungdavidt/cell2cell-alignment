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
    
    disp(['stiching slice ',num2str(nn)])
    n_raw_im = n_raw_im+1;

    in_slice=filt_neurons.slice==nn;
    %figure; hold on
    %plot(filt_neurons.pos(in_slice&pass_qc,1),-filt_neurons.pos(in_slice&pass_qc,2), '.r');


    % find the corresponding HYB
    fov_names = unique(filt_neurons.fov_names(filt_neurons.fov(in_slice&pass_qc)));
    fov_id = unique(filt_neurons.fov(in_slice&pass_qc));

    im_tmp = uint16(0);
    im_tmp2 = uint16(0);

    for i_fov = 1:length(fov_names)

        disp(['FOV ',num2str(i_fov)])

        hyb_file = ['.\hyb\',fov_names{i_fov},'\alignedn2vhyb01.tif'];
        im_hyb2 = imread(hyb_file);
        im_hyb = imread(hyb_file,'Index',4);

        f_cell_mask = ['.\hyb\',fov_names{i_fov},'\cellmask.mat'];
        load(f_cell_mask)
        
        i_cell = find(filt_neurons.fov==fov_id(i_fov));     % find cells in the FOV
    
        keyboard
        
        % find offsets based position in the stitched image ('filt_neurons.pos') and raw HYB image ('filt_neurons.pos40x')
        % need to use double or will run out machine precision
        b = regress(double(filt_neurons.pos(i_cell,1))*2,[ones(size(filt_neurons.pos40x(i_cell,1))) double(filt_neurons.pos40x(i_cell,1))]);
        x_offset_new = round(b(1));
        b = regress(double(filt_neurons.pos(i_cell,2))*2,[ones(size(filt_neurons.pos40x(i_cell,2))) double(filt_neurons.pos40x(i_cell,2))]);
        y_offset_new = round(b(1));

        % take the max-projection in the overlap area
        im_tmp(y_offset_new+1:y_offset_new+3200, x_offset_new+1:x_offset_new+3200,2) = im_hyb;
        im_tmp=max(im_tmp,[],3);

        im_tmp2(y_offset_new+1:y_offset_new+3200, x_offset_new+1:x_offset_new+3200,2) = im_hyb2;
        im_tmp2=max(im_tmp2,[],3);

        %figure
        %image(im_tmp); hold on
        %plot(filt_neurons.pos(xxx,1)*2,filt_neurons.pos(xxx,2)*2,'.r');
        %figure
        %image(im_stitch); hold on
        %plot(filt_neurons.pos(xxx,1)/5,filt_neurons.pos(xxx,2)/5,'.r');
        %keyboard

    end

%     im_tmp(im_tmp>120)=120;
    im_tmp3 = int16(double(im_tmp).^3/50);%*40;
    %im_tmp3(im_tmp3<40)=0;
    figure
    image(im_tmp3); hold on
    figure
    image(im_tmp2); hold on
    %plot(filt_neurons.pos(in_slice&pass_qc,1)*2,filt_neurons.pos(in_slice&pass_qc,2)*2, '.r');
    keyboard

    raw_image_stack{n_raw_im,1} = im_tmp;
    raw_image_stack{n_raw_im,2} = [filt_neurons.pos(in_slice&pass_qc,1)*2 filt_neurons.pos(in_slice&pass_qc,2)*2];

end

save stitched_HYB raw_image_stack


% plot stitched sections
for i_im = 3:length(raw_image_stack)
    im1 = raw_image_stack{i_im-1,1};      % fixed
    im2 = raw_image_stack{i_im,1};        % moving

    % trim size
    x_offset = min(raw_image_stack{i_im-1,2}(:,1))
    y_offset = min(raw_image_stack{i_im-1,2}(:,2))
    im1 = im1((1:12000)+y_offset, (1:9000)+x_offset);
    x_offset = min(raw_image_stack{i_im,2}(:,1))
    y_offset = min(raw_image_stack{i_im,2}(:,2))
    im2 = im2((1:12000)+y_offset, (1:9000)+x_offset);

    figure
    subplot(1,2,1); image(im1);
    subplot(1,2,2); image(im2);
    keyboard

end


