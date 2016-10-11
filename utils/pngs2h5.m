function [h5out_mean, h5out_std, empty_vol, ...
          h5out_kl_divergence, p_hist] = ...
      pngs2h5(base_dir, num_z, fn, do_normalize, ...
              do_convert_coords, z_offset, img_prefix, ...
              bg_vals_to_nan, q_hist)
% PNGS2H5 convert image data in pngs to h5 file
% [h5out_mean, h5_out_std] = PNGS2H5(...
%      base_dir, num_z, fn, do_normalize, ...
%      do_convert_coords, z_offset, img_prefix)
%
% do_normalize, do_convert_coords default to 1 if not specified
% if num_z < 0, all matching images used
%
% example:
%   base_dir = ...
%     '/groups/flyem/data/viren_toufiq_comparison/trvol-250-1';
%   num_z    = 250;
%   fn       = ...
%     ['/groups/flyem/data/viren_toufiq_comparison/' ...
%      'trvol-250-1-dawmr/img_normalized.h5'];
%
% notes:
%   for now, assume we can read entire volume into memory as double
%   convert coordinates using flipud(rot90()) to fix xy indexing

  if(~exist('do_normalize','var') || isempty(do_normalize))
    do_normalize = true;
  end
  if(~exist('do_convert_coords','var') || ...
     isempty(do_convert_coords))
    do_convert_coords = true;
  end
  if(~exist('z_offset','var') || isempty(z_offset))
    z_offset = 0;
  end
  if(~exist('img_prefix','var') || isempty(img_prefix))
    img_prefix = 'img';
  end

  h5out_kl_divergence = [];

  img_dir = [base_dir];% '/grayscale_maps'];

  if(num_z < 0)
    img_fns = dir(sprintf('%s/%s.*.png', img_dir, img_prefix));
    num_z   = length(img_fns);
    assert(num_z > 0, 'DAWMRLIB:AssertionFailed', ...
           'no images found');
    img     = imread(sprintf('%s/%s', img_dir, img_fns(1).name));
  else
    img_fns = [];
    img     = imread(sprintf('%s/%s.%05d.png', ...
                             img_dir, img_prefix, z_offset));
  end

  if(do_convert_coords)
    % x and y swapped in pngs
    h   = size(img,2);
    w   = size(img,1);
  else
    h   = size(img,1);
    w   = size(img,2);
  end

  h5out = zeros(h,w,num_z, 'double');

  for ii=1:num_z
    if(isempty(img_fns))
      img_fn = sprintf('%s/%s.%05d.png', ...
                       img_dir, img_prefix, ii-1+z_offset);
    else
      img_fn = sprintf('%s/%s', img_dir, img_fns(ii).name);
    end

    img    = double(imread(img_fn));

    if(do_convert_coords)
      % convert coordinates
      img    = flipud(rot90(img));
    end

    h5out(:,:,ii) = img;
  end

  if(exist('bg_vals_to_nan','var') && ...
     ~isempty(bg_vals_to_nan))
    for bg_val = bg_vals_to_nan(:)'
      h5out(h5out == bg_val) = NaN;
    end
  end

  empty_vol = nnz(~isnan(h5out)) == 0;
  if(empty_vol)
    h5out_mean = NaN;
    h5out_std  = NaN;
    return
  end

  if(exist('q_hist','var') && ~isempty(q_hist))
    p_hist = image2hist(h5out);
    h5out_kl_divergence = kl_divergence_hist(p_hist, q_hist);
  else
    p_hist = [];
    h5out_kl_divergence = [];
  end

  if(~isequal(do_normalize, 0))
    if(do_normalize==1) % self normalize
      h5out_mean = mean(h5out(~isnan(h5out)));
      h5out_std  = std( h5out(~isnan(h5out)));
    else
      assert(...
        length(do_normalize)==2, ...
        'DAWMRLIB:AssertionFailed', ...
        'do_normalize should either be true or [mn std] to use');

      h5out_mean = do_normalize(1);
      h5out_std  = do_normalize(2);
    end
    fprintf('normalizing by mean: %d / std: %d\n', ...
            h5out_mean, h5out_std);
    h5out = (h5out - h5out_mean) ./ h5out_std;
  else
    h5out_mean = [];
    h5out_std  = [];
  end

  h5out = single(h5out);

  if(~isempty(fn))
    if(exist(fn, 'file'))
      delete(fn);
    end

    h5create(fn, '/main', size(h5out), ...
             'Datatype', 'single', ...
             'ChunkSize', [50 50 50], ...
             'Deflate', 4, ...
             'Shuffle', 1);
    h5write(fn, '/main', h5out);
  end
end
