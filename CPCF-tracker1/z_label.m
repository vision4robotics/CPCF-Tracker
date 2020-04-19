function z_label = z_label(response,rate,filter_sz_cell,num_feature_blocks,base_target_sz,feature_cell_sz,params)
zf = cell(numel(num_feature_blocks), 1);
mag_min = 0.02;
mag_max = 0.2;
dpmr_value = dpmr(response,rate);
if dpmr_value>500 
    dpmr_value = 500;
end
if dpmr_value<0
    dpmr_value = 0;
end
    for i = 1:num_feature_blocks
        sz = filter_sz_cell{i};
        output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * params.rc_sigma_factor;
        rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
        cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
        [rs, cs]     = ndgrid(rg,cg);
        %     z            = 0.02*exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
        %0.02~0.2
        magnitude = mag_min+(mag_max-mag_min)*(dpmr_value/500);
        z            = magnitude*exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
        zf          = fft2(z); 
    end
z_label = zf;
end