function results = tracker_HOGCN(params)
%% Initialization
% Get sequence info
learning_rate = params.learning_rate;
admm_gamma = params.admm_gamma;

[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end
% Init position
pos = seq.init_pos(:)';
% context position
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;
rate  = params.rate;
% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;


% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

all_resp_corr = [];



% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'exact');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = unique(feature_info.data_sz, 'rows', 'stable');
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');
num_feature_blocks = size(feature_sz, 1);

small_filter_sz{1} = floor(base_target_sz/(feature_cell_sz(1,1)));

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = mat2cell(feature_sz, ones(1,num_feature_blocks), 2);
filter_sz = feature_sz;
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

filter_sz_cell_ours{1} = filter_sz_cell{1}; 

% Initialize previous and current response maps
M_prev= zeros(filter_sz_cell{1});
M_curr = zeros(filter_sz_cell{1});
response = zeros(filter_sz_cell{1});
M_prev_plot = zeros(filter_sz_cell{1});
M_curr_plot = zeros(filter_sz_cell{1});

% The size of the label function DFT. Equal to the maximum filter size
[output_sz_hand, k1] = max(filter_sz, [], 1);

output_sz = output_sz_hand;

k1 = k1(1);
% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% Construct the Gaussian label function
yf = cell(numel(num_feature_blocks), 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * params.output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf          = fft2(y);  
end

% Define spatial regularization windows
reg_window = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    reg_scale = floor(base_target_sz/params.feature_downsample_ratio(i));
    use_sz = filter_sz_cell{i};    
    reg_window{i} = ones(use_sz) * params.reg_window_max;
    range = zeros(numel(reg_scale), 2);
    
   % determine the target center and range in the regularization windows
    for j = 1:numel(reg_scale)
        range(j,:) = [0, reg_scale(j) - 1] - floor(reg_scale(j) / 2);
    end
    center = floor((use_sz + 1)/ 2) + mod(use_sz + 1,2);
    range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
    range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));
    
    reg_window{i}(range_h, range_w) = params.reg_window_min;
end


% Initialize for label z
z = zeros(filter_sz_cell{1});
zf = cell(numel(num_feature_blocks), 1);
mag_min = params.mag_min;
mag_max = params.mag_max;
magnitude = params.magnitude;
num_frames = params.frames;
all_psr = zeros(num_frames,1);
all_Rmax = zeros(num_frames,1);
all_psrm = zeros(num_frames,1);
rc_sigma_factor = params.rc_sigma_factor;


% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', feature_sz_cell, 'uniformoutput', false);

% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
% parameters for scale estimation
scale_sigma = sqrt(params.num_scales) * params.scale_sigma_factor;
ss = (1:params.num_scales) - ceil(params.num_scales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));
if mod(params.num_scales,2) == 0
    scale_window = single(hann(params.num_scales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(params.num_scales));
end
ss = 1:params.num_scales;
scaleFactors = params.scale_step.^(ceil(params.num_scales/2) - ss);
if params.scale_model_factor^2 * prod(params.init_sz) > params.scale_model_max_area
    params.scale_model_factor = sqrt(params.scale_model_max_area/prod(params.init_sz));
end

if prod(params.init_sz) > params.scale_model_max_area
    params.scale_model_factor = sqrt(params.scale_model_max_area/prod(params.init_sz));
end
scale_model_sz = floor(params.init_sz * params.scale_model_factor);

% set maximum and minimum scales
min_scale_factor = params.scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(params.scale_step));
max_scale_factor = params.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(params.scale_step));


seq.time = 0;

% Define the learning variables
% h_current_key = cell(num_feature_blocks, 1);
cf_f = cell(num_feature_blocks, 1);

% Allocate
%scores_fs_feat = cell(1,1,num_feature_blocks);
scores_fs_feat = cell(1,1,3);
%% Main loop here
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end

    tic();
    
    % Target localization step
    
    % Do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
%             sample_scale = currentScaleFactor*scaleFactors;
            xt = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
            % Do windowing of features  ¼ÓÓàÏÒ´°
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf{k1}), 3));
            scores_fs_sum = scores_fs_feat{k1};
            for k = block_inds
                scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
            end
             
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            
            responsef_padded = resizeDFT2(scores_fs, output_sz);
            response = ifft2(responsef_padded, 'symmetric');
            %
            
            %
            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, output_sz);

            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor;            
%             scale_change_factor = scaleFactors(sind);
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            % lfl: SCALE SPACE SEARCH
            xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
            xsf = fft(xs,[],2);
            scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + params.scale_lambda) ));            
            % find the maximum scale response
            recovered_scale = find(scale_response == max(scale_response(:)), 1);
            % update the scale
            currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end            

            M_curr = fftshift(response(:,:,sind));
            max_M_curr = max(M_curr(:));
            
            [id_ymax_curr, id_xmax_curr] = find(M_curr == max_M_curr);
            
            id_y_center = round( size(M_curr,1)/2) ;
            id_x_center = round( size(M_curr,1)/2);
          
            shift_y = id_ymax_curr - id_y_center;
            shift_x = id_xmax_curr - id_x_center;
            sz_shift_y = size(shift_y);
            sz_shift_x = size(shift_x);
            if(sz_shift_y(1) > 1)
                shift_y = shift_y(1);
            end
            if(sz_shift_x(1) > 1)
                shift_x = shift_x(1);
            end
            M_curr= circshift(M_curr,shift_y,1);
            M_curr= circshift(M_curr,shift_x,2); 
            iter = iter + 1;
        end
       
    end
        
    

    %% Model update step
    % extract image region for training sample
    sample_pos = round(pos);
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
    % do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    % compute the fourier series
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    % train the CF model for each feature
    for k = 1: 1
        
             if (seq.frame == 1)
                model_xf = xlf{k};
                psrm = 50;
                magnitude = 0;
            else
                %Limit the value of psr
                model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xlf{k});
                psr_value = psr(M_curr,rate);
                Rmax = max(M_curr(:));
                beta = 100;
                psrm = psr_value + beta * Rmax;
                all_psr(seq.frame) = psr_value;
                all_Rmax(seq.frame) = Rmax;
                all_psrm(seq.frame)  = psrm;
                if psrm>50 
                    psrm = 50;
                end
                if psrm<0
                    psrm = 0;
                end 
                magnitude = mag_min+(mag_max-mag_min)*(psrm/50);
            end
            
            g_f = single(zeros(size(xlf{k})));
            h_f = g_f;
            l_f = g_f;
            mu    = 1;
            betha = 10;
            mumax = 100 ;  
            i = 1;
            params.admm_iterations = 2;
            zf = conj(yf) .* yf ;
            z = magnitude * ifft2(zf);
            zf = fft2(z);
           
            T = prod(filter_sz_cell_ours{k});
            S_xx = sum(conj(model_xf) .* model_xf, 3);
            all1 = ones(filter_sz_cell_ours{k});
            A = mu ./ (all1 + admm_gamma*(conj(M_curr).*M_curr));          
            B = S_xx + T * A;
            C = 1./A;
            S_lx = sum(conj(model_xf) .* l_f, 3);
            S_hx = sum(conj(model_xf) .* h_f, 3);
            model_xmf = bsxfun(@times, M_curr,model_xf);
            S_xm = sum(conj(model_xf) .* model_xmf, 3);
            
            % ADMM solving process
            while (i <= params.admm_iterations)
                             
                g_f = bsxfun(@times, (A / mu) , ( ((bsxfun(@times, (C ./ T) , bsxfun(@times, yf, model_xf)) + bsxfun(@times, (admm_gamma * C) , bsxfun(@times, zf , model_xmf)) - (bsxfun(@times, C ,l_f)) + (bsxfun(@times, mu*C, h_f)) - ...
                    bsxfun(@rdivide,((bsxfun(@times, (C ./ T) , bsxfun(@times, model_xf, (S_xx .* yf)))) + bsxfun(@times,(admm_gamma * C) , bsxfun(@times, model_xf, (S_xm .* zf))) - (bsxfun(@times , C , bsxfun(@times, model_xf, S_lx))) + bsxfun(@times,(mu*C),(bsxfun(@times, model_xf, S_hx)))) , B)))));
   
                %   solve for H
                h_f = fft2(argmin_g(reg_window{k}, mu, real(ifft2(mu * g_f+ l_f)), h_f));

                %   update L
                l_f = l_f + (mu * (g_f - h_f));
                cf_f{k} = g_f;
                %   update mu- betha = 10.
                mu = min(betha * mu, mumax);  
                i = i+1;
            end
    end
    if(seq.frame == 1)
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xlf{k1}), 3));
            scores_fs_sum = scores_fs_feat{k1};
            for k = block_inds
                scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xlf{k}), 3));
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
            end
             
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            
            responsef_padded = resizeDFT2(scores_fs, output_sz);
            response = ifft2(responsef_padded, 'symmetric');
                       
            M_curr = fftshift(response);
            max_M_curr = max(M_curr(:));
            [id_ymax_curr,id_xmax_curr]= find(M_curr == max_M_curr);            
    else
            M_prev_plot = M_prev;
            M_curr_plot = M_curr;
            M_prev = M_curr;
            max_M_prev = max_M_curr;
            id_ymax_prev = id_ymax_curr;
            id_xmax_prev = id_xmax_curr;
    end
    
    %% Upadate Scale
    xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    
    if seq.frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - params.learning_rate_scale) * sf_den + params.learning_rate_scale * new_sf_den;
        sf_num = (1 - params.learning_rate_scale) * sf_num + params.learning_rate_scale * new_sf_num;
    end
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %fprintf('target_sz: %f, %f \n', target_sz(1), target_sz(2));
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    seq.time = seq.time + toc();
    
    %% Visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
           
%     code for visualization
        all_psr(seq.frame) = psr(M_curr_plot,rate);
        M_prevf = fft2(M_prev_plot);
        M_currf = fft2(M_curr_plot);
        resp_corr = fftshift(ifft2(conj(M_currf) .* M_prevf));
        all_resp_corr(end+1) = max(max(resp_corr));

        drawnow   
        
        imshow(im_to_show);
        hold on;
        rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
        text(10, 10, [int2str(seq.frame) '/'  int2str(seq.len)], 'color', [0 1 1]);
        hold off;
        axis off;axis image;
                    
        drawnow
    end
end

[~, results] = get_sequence_results(seq);

disp(['fps: ' num2str(results.fps)])

