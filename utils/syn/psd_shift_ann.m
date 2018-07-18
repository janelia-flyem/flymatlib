function pp = psd_shift_ann(tt, pp, dvid_conn, seg_name, ...
                            window_r, erode_r, z_r, ...
                            cube_seg_fn, seg_offset)
% PSD_SHIFT_ANN shift psd annotations inside, grouped z-planes
%   point annotations are shifted away from segment boundaries,
%   and grouped into a minimal number of z-planes
% pp = PSD_SHIFT_ANN(tt, pp, dvid_conn, seg_name, ...
%                    window_r, erode_r, z_r)

  % setting default values
  if(~exist('window_r','var') || isempty(window_r))
    window_r = 60;
  end
  if(~exist('erode_r','var') || isempty(erode_r))
    erode_r = 6;
  end
  if(~exist('z_r','var') || isempty(z_r))
    z_r = 6;
  end

  use_dvid = true;
  if(isempty(dvid_conn))
    use_dvid = false;
    local_seg = read_image_stack(cube_seg_fn);
  end

  [dx,dy,dz] = ndgrid(-window_r:window_r, ...
                      -window_r:window_r, ...
                      -window_r:window_r);
  dist_tbar = sqrt(dx.^2 + dy.^2 + dz.^2);
  window_sz = (2*window_r + 1)*ones(1,3);
  erode_flt = cell(1,erode_r);
  for ee = 1:erode_r
    erode_flt{ee} = fml_set_filter(ee);
  end
  opts = optimoptions('intlinprog','Display','off');


  %% going through each synapse
  for ii=1:length(pp)
    if(mod(ii,500)==0),fprintf('[%d]',ii);end
    if(isempty(pp{ii})), continue, end

    % move to local coordinates, 1-indexing
    vol_start = tt(1:3,ii)-window_r;
    pp_loc = bsxfun(@minus, pp{ii}(1:3,:), vol_start) + 1;

    % get local segmentation
    if(use_dvid)
      ss = dvid_conn.get_segmentation(...
          vol_start', window_sz, [], seg_name);
    else
      xx = vol_start(2) - seg_offset(2);
      yy = vol_start(1) - seg_offset(1);
      zz = vol_start(3) - seg_offset(3);
      ss = local_seg(xx+(1:window_sz(1)),...
                     yy+(1:window_sz(2)),...
                     zz+(1:window_sz(3)));
    end

    n_psd = size(pp_loc,2);
    seg_valid_ind = cell(1, n_psd);
    Amat = zeros(n_psd,window_sz(3));
    bmat = -ones(n_psd,1);
    ff   = ones(window_sz(3),1);
    lb   = zeros(window_sz(3),1);
    ub   = ones( window_sz(3),1);
    %% going through each psd
    for jj=1:n_psd
      % get seg id
      seg_id = ss(pp_loc(2,jj),pp_loc(1,jj),pp_loc(3,jj));
      % erode, find closest point
      for ee = erode_r:-1:1
        seg_mask = double(~(convolution3D_FFTdomain( ...
            double(ss~=seg_id), erode_flt{ee}) > 1e-4));
        if(nnz(seg_mask)>0), break; end
      end
      if(nnz(seg_mask)==0)
        seg_mask = double(ss==seg_id);
      end
      seg_dist_tbar = dist_tbar;
      seg_dist_tbar(~seg_mask) = Inf;

      min_dist = min(seg_dist_tbar(:));
      dist_thresh = min_dist + z_r;

      % get set of candidate indices
      seg_valid = double(seg_dist_tbar <= dist_thresh);
      seg_valid_ind{jj} = find(seg_valid);

      dist_cp = seg_dist_tbar;
      dist_cp(dist_cp > dist_thresh) = dist_thresh;
      ff = ff + squeeze(min(min(dist_cp,[],1),[],2)) ...
           / (2*dist_thresh*n_psd);

      % set up A matrix
      z_valid = squeeze(sum(sum(seg_valid,1),2)>0);
      Amat(jj,z_valid) = -1;
    end

    %% run constrained integer opt to find z-planes to use
    z_to_use = logical(intlinprog(ff,1:window_sz(3),Amat,bmat,...
                                  [],[],lb,ub,opts));

    %% set new assignment for each psd
    for jj=1:n_psd
      seg_valid = zeros(size(ss));
      seg_valid(seg_valid_ind{jj}) = 1;
      % zero out non-selected z-planes
      seg_valid(:,:,~z_to_use) = 0;

      seg_dist_tbar = dist_tbar;
      seg_dist_tbar(~seg_valid) = Inf;
      [~,closest_loc] = min(seg_dist_tbar(:));
      [cx,cy,cz] = ind2sub(size(ss), closest_loc);

      % write out new location in global coordinates, 0-indexing
      pp{ii}(1:3,jj) = [cy; cx; cz] - 1 + ...
          tt(1:3,ii) - window_r;
    end

    % pp_loc
    % bsxfun(@minus, pp{ii}(1:3,:), tt(1:3,ii) - window_r - 1)

  end
  fprintf('\n');

end
