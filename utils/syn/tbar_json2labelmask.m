function tbar_json2labelmask(fn_json, fn_out_prefix, ...
                             vol_sz, radius_use, border, ...
                             z_offset, ...
                             fn_image, image_thresh, ...
                             radius_ign, ...
                             add_center_out)
% TBAR_JSON2LABELMASK convert tbar json annotations to groundtruth
%   hdf5 files for training dawmr or CNN
% TBAR_JSON2LABELMASK(fn_json, fn_out_prefix, ...
%   vol_sz, radius_to_use, border, z_offset, ...
%   fn_image, image_thresh, radius_ign)
%
% fn_json          json filename
% fn_out_prefix    output filename prefix
% vol_sz           total substack volume size
% radius_to_use    object radius to set positive voxel labels
% border           zero out labels close to border (mask)
% z_offset         account for z offset in json coordinates
% fn_image         image filename if using image thresholding
% image_thresh     image threshold for positive voxel labels
% radius_ign       ignore region radius, neither pos or neg labels
% add_center_out   output layers for object center prediction


  if(~exist('z_offset','var'))
    z_offset = [];
  end
  if(~exist('radius_ign','var'))
    radius_ign = 0;
  end
  if(~exist('add_center_out','var') || isempty(add_center_out))
    add_center_out = false;
  end

  radius_max = max([radius_use, radius_ign]);

  %% set labels

  if(~add_center_out)
    labels = zeros(vol_sz + 2*radius_max);
  else
    labels = zeros([vol_sz + 2*radius_max, 4]);
  end

  % set labels, using radius of cube
  if(ischar(fn_json)) % read from file
    locs = tbar_json2locs(fn_json, z_offset);
    % switch coordinate ordering by
    %   swapping x and y, and re-orienting both

    % skip for python use, json already for python
    %locs      = locs([2 1 3], :);
    %locs(1,:) = vol_sz(1) - locs(1,:) - 1;
  else % passed in and already in local matlab 0-based coordinates
    locs = fn_json;
  end
  % account for padding and 0-based indexing
  locs      = locs + 1 + radius_max;

  [flt, flt_dist] = fml_set_filter(radius_use);

  for ii=1:size(locs,2)
    cx = locs(1,ii); cy = locs(2,ii); cz = locs(3,ii);

    labels(cx-radius_use:cx+radius_use, ...
           cy-radius_use:cy+radius_use, ...
           cz-radius_use:cz+radius_use,1) = ...
      labels(cx-radius_use:cx+radius_use, ...
             cy-radius_use:cy+radius_use, ...
             cz-radius_use:cz+radius_use,1) | flt;
  end

  if(add_center_out)
    % output prediction for closest tbar
    label_sz   = size(labels);
    label_dist = Inf*ones(label_sz(1:3));
    local_sz   = (2*radius_use+1)*ones(1,3);

    for ii=1:size(locs,2)
      cx = locs(1,ii); cy = locs(2,ii); cz = locs(3,ii);

      curr_dist = label_dist(cx-radius_use:cx+radius_use, ...
                             cy-radius_use:cy+radius_use, ...
                             cz-radius_use:cz+radius_use);

      use_mask  = (flt_dist < curr_dist);
      new_dist  = min(flt_dist, curr_dist);

      label_dist(cx-radius_use:cx+radius_use, ...
                 cy-radius_use:cy+radius_use, ...
                 cz-radius_use:cz+radius_use) = new_dist;

      for cc=1:3
        repmat_sz      = local_sz;
        repmat_sz(cc)  = 1;
        reshape_sz     = ones(1,3);
        reshape_sz(cc) = 2*radius_use+1;
        local_center   = repmat(reshape(...
          radius_use:-1:-radius_use, reshape_sz), ...
                                repmat_sz);

        labels(cx-radius_use:cx+radius_use, ...
               cy-radius_use:cy+radius_use, ...
               cz-radius_use:cz+radius_use,1+cc) = ...
            labels(cx-radius_use:cx+radius_use, ...
                   cy-radius_use:cy+radius_use, ...
                   cz-radius_use:cz+radius_use,1+cc) .* ...
            ~use_mask + local_center .* use_mask;
      end

    end
  end


  labels = labels(radius_max+1:end-radius_max, ...
                  radius_max+1:end-radius_max, ...
                  radius_max+1:end-radius_max, :);

  % apply image thresh if requested
  if(exist('fn_image','var') && exist('image_thresh','var') && ...
     ~isempty(image_thresh))
    im     = read_image_stack(fn_image);
    labels(:,:,:,1) = double(labels(:,:,:,1) & ...
                             (im <= image_thresh(1)));
  end


  %% set mask

  mask   = ones(vol_sz + 2*radius_max);

  % set ignore region if requested
  if(radius_ign > 0)
    flt_ign = 1-fml_set_filter(radius_ign);

    for ii=1:size(locs,2)
    cx = locs(1,ii); cy = locs(2,ii); cz = locs(3,ii);

    mask(cx-radius_ign:cx+radius_ign, ...
         cy-radius_ign:cy+radius_ign, ...
         cz-radius_ign:cz+radius_ign) = ...
      mask(cx-radius_ign:cx+radius_ign, ...
           cy-radius_ign:cy+radius_ign, ...
           cz-radius_ign:cz+radius_ign) & flt_ign;
    end
  end

  mask = mask(radius_max+1:end-radius_max, ...
              radius_max+1:end-radius_max, ...
              radius_max+1:end-radius_max);

  if(radius_ign > 0)
    % don't ignore any positive labels
    mask = double(mask | labels(:,:,:,1));

    if(exist('image_thresh','var') && length(image_thresh)>1)
      % don't ignore any easy negatives
      mask = double(mask | (im >= image_thresh(2)));
    end
  end

  % account for border
  mask(1:border,:,:) = 0;
  mask(:,1:border,:) = 0;
  mask(:,:,1:border) = 0;
  mask(end-border+1:end,:,:) = 0;
  mask(:,end-border+1:end,:) = 0;
  mask(:,:,end-border+1:end) = 0;

  labels = int8(labels);
  mask   = int8(mask);

  fn_labels = sprintf('%slabels.h5', fn_out_prefix);
  fn_mask   = sprintf('%smask.h5',   fn_out_prefix);

  if(exist(fn_labels, 'file'))
    delete(fn_labels);
  end
  if(exist(fn_mask, 'file'))
    delete(fn_mask);
  end

  chunk_size_labels = [50 50 50];
  if(add_center_out)
    chunk_size_labels = [chunk_size_labels 1];
  end
  h5create(fn_labels, '/main', size(labels), ...
           'Datatype', 'int8', ...
           'ChunkSize', chunk_size_labels, ...
           'Deflate', 4, ...
           'Shuffle', 1);
  h5create(fn_mask, '/main', size(mask), ...
           'Datatype', 'int8', ...
           'ChunkSize', [50 50 50], ...
           'Deflate', 4, ...
           'Shuffle', 1);

  h5write(fn_labels, '/main', labels);
  h5write(fn_mask,   '/main', mask);

end
