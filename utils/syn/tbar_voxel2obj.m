function tbar_voxel2obj(...
    fn_voxel, jsonoutputfile, thd, ave_radius, obj_min_dist, ...
    vol_offset, buffer_sz, vol_tot_2, from_ilastik)
% TBAR_VOXEL2OBJ tbar produce pointwise annotations to json file
%   from dense voxel-wise predictions, by applying smoothing
%   and non-maxima suppression (nms)
%
%   fn_voxel   filename of voxel predictions
%   json_out   filename for json output
%   thd        confidence threshold
%   ave_radius   smoothing radius
%   obj_min_dist   minimum distance between tbars (nms)
%   vol_offset     volume offset in global coordinates
%   buffer_sz      size of buffer to ignore
%   vol_tot_2      y (2nd) dimension total size (for conversion)
%   from_ilastik   if fn_voxel was produced by ilastik

  if(~exist('vol_offset','var') || isempty(vol_offset))
    vol_offset = [0 0 0];
  else
    if(isscalar(vol_offset))
      vol_offset = [0 0 vol_offset];
    end
  end

  if(~exist('from_ilastik','var') || isempty(from_ilastik))
    from_ilastik = false;
  end

  pad_radius = max(ave_radius, obj_min_dist);
  if(~from_ilastik)
    vv_sz = get_h5_size(fn_voxel);
  else
    hh    = h5info(fn_voxel);
    vv_sz = hh.Groups.Datasets.Dataspace.Size(2:4);
  end
  vv_pd      = zeros(vv_sz(1:3) + 2*pad_radius);
  vv_pd_sz   = size(vv_pd);

  if(~exist('vol_tot_2','var') || isempty(vol_tot_2))
    vol_tot_2 = vv_sz(2);
  end

  % pad volume for easier indexing
  if(~from_ilastik)
    vv_pd(pad_radius+1:end-pad_radius, ...
          pad_radius+1:end-pad_radius, ...
          pad_radius+1:end-pad_radius) = ...
      read_image_stack(fn_voxel);
  else
    vv_temp = hdf5read(fn_voxel,'volume/predictions');
    vv_temp = squeeze(vv_temp(1,:,:,:,1));
    vv_temp = permute(vv_temp, [2 1 3]);

    vv_pd(pad_radius+1:end-pad_radius, ...
          pad_radius+1:end-pad_radius, ...
          pad_radius+1:end-pad_radius) = ...
      vv_temp;

    clear vv_temp
  end

  % get averaging filter, do convolution
  nan_mask = isnan(vv_pd);
  if(nnz(nan_mask)>0)
    vv_pd(nan_mask) = 0;
  end
  if(~isempty(ave_radius) && ave_radius > 0)
    ave_flt = fml_set_filter(ave_radius);
    ave_flt = ave_flt ./ sum(ave_flt(:));
    vv_pd   = convolution3D_FFTdomain(vv_pd, ave_flt);
  end
  if(nnz(nan_mask)>0)
    vv_pd(nan_mask) = NaN;
  end

  % eliminate buffer_sz from consideration
  if(~exist('buffer_sz','var') || isempty(buffer_sz))
    buffer_sz = 0;
  end
  if(isscalar(buffer_sz))
    buffer_sz = buffer_sz*ones(1,3);
  end

  vv_pd([1:pad_radius, end-pad_radius+1:end],:,:) = 0;
  vv_pd(:,[1:pad_radius, end-pad_radius+1:end],:) = 0;
  vv_pd(:,:,[1:pad_radius, end-pad_radius+1:end]) = 0;

  % select top points
  % quantile_cutoff = 0.975;
  % thresh = quantile(vv_pd(:), quantile_cutoff);
  thresh = 0.3;
  inds   = find(vv_pd > thresh);

  % get distance filter, initialize is_valid
  dist_flt = logical(fml_set_filter(obj_min_dist));
  is_valid = true(vv_pd_sz);

  % select top value, set as new object location, update is_valid
  locs = zeros(4,0);
  while(~isempty(inds))
    vals                = vv_pd(inds);
    [max_val, max_ind]  = max(vals);

    if(max_val <= 0)
      fprintf('stopping early\n');
      break
    end

    [xx, yy, zz]    = ind2sub(vv_pd_sz, inds(max_ind));
    locs(1:3,end+1) = [xx;yy;zz] - pad_radius; %#ok<AGROW>
    locs(4,end)     = vals(max_ind);

    is_valid(xx-obj_min_dist:xx+obj_min_dist, ...
             yy-obj_min_dist:yy+obj_min_dist, ...
             zz-obj_min_dist:zz+obj_min_dist) = ...
      is_valid(xx-obj_min_dist:xx+obj_min_dist, ...
               yy-obj_min_dist:yy+obj_min_dist, ...
               zz-obj_min_dist:zz+obj_min_dist) & ...
      ~dist_flt;

    inds(~is_valid(inds)) = [];
  end

  % remove points in buffer_sz
  in_buffer = ...
      locs(1,:) <= buffer_sz(1) | ...
      locs(1,:) >  vv_sz(1) - buffer_sz(1) | ...
      locs(2,:) <= buffer_sz(2) | ...
      locs(2,:) >  vv_sz(2) - buffer_sz(2) | ...
      locs(3,:) <= buffer_sz(3) | ...
      locs(3,:) >  vv_sz(3) - buffer_sz(3);
  locs(:,in_buffer) = [];

  % write out json file
  cc_ctrs = locs(:, locs(4,:) >= thd);
  % swap matlab to dvid/raveler x,y ordering
  cc_ctrs        = cc_ctrs([2 1 3 4], :);
  % shift to global coordinates and 0-based indexing
  cc_ctrs(1:3,:) = bsxfun(@plus, cc_ctrs(1:3,:), ...
                          vol_offset') - 1;
  % shift to raveler coordinates
  cc_ctrs(2,:)   = vol_tot_2 - cc_ctrs(2,:) - 1;

  tbar_json_write(jsonoutputfile, cc_ctrs);
end
