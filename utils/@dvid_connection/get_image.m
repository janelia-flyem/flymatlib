function [im_mean, im_std, empty_vol] = get_image(...
    this, vol_start, vol_sz, image_fn, ...
    do_normalize, bg_vals_to_nan)

  if(exist(image_fn, 'file')), delete(image_fn); end
  max_tries = 5;
  dvid_cmd  = ...
      sprintf(['%s get ' ...
               '"%s/api/node/%s/grayscale/raw/0_1_2/' ...
               '%d_%d_%d/%d_%d_%d?throttle=on&%s" > %s'], ...
              this.http_cmd, ...
              this.machine_name, this.repo_name, ...
              vol_sz(1),    vol_sz(2),    vol_sz(3),    ...
              vol_start(1), vol_start(2), vol_start(3), ...
              this.user_string, image_fn);
  st = system(dvid_cmd);
  num_tries = 1;
  while(st ~= 0 && num_tries < max_tries)
    fprintf('status code = %d ,', st);
    if(st ~= 5) % not 503 unavailable from throttling
      num_tries = num_tries+1;
    end
    pause(30);

    fprintf('retrying...\n');
    st = system(dvid_cmd);
  end
  if(st ~= 0)
    error('error connecting to dvid: %s/%s/', ...
          this.machine_name, this.repo_name);
  end

  fid = fopen(image_fn);
  assert(fid>0, 'FML:AssertionFailed', ...
         sprintf('could not read file: %s', image_fn));

  im = fread(fid, Inf, 'uint8=>double');

  fclose(fid);
  delete(image_fn);

  % reshape and switch from DVID to Matlab indexing order
  im = reshape(im, vol_sz);
  im = permute(im, [2 1 3]);

  if(exist('bg_vals_to_nan','var') && ...
     ~isempty(bg_vals_to_nan))
    for bg_val = bg_vals_to_nan(:)'
      im(im == bg_val) = NaN;
    end
  end

  empty_vol = nnz(~isnan(im)) == 0;
  if(empty_vol)
    im_mean = NaN;
    im_std  = NaN;
    return
  end

  if(~isequal(do_normalize, 0))
    if(do_normalize==1) % self normalize
      im_mean = mean(im(~isnan(im)));
      im_std  = std( im(~isnan(im)));
    else
      assert(...
          length(do_normalize)==2, ...
          'FML:AssertionFailed', ...
          'do_normalize should either be true or [mn std] to use');

      im_mean = do_normalize(1);
      im_std  = do_normalize(2);
    end
    fprintf('normalizing by mean: %d / std: %d\n', ...
            im_mean, im_std);
    im = (im - im_mean) ./ im_std;
  else
    im_mean = [];
    im_std  = [];
  end

  im = single(im);
  chunk_sz = min([50 50 50], vol_sz);

  h5create(image_fn, '/main', size(im), ...
           'Datatype', 'single', ...
           'ChunkSize', chunk_sz, ...
           'Deflate', 4, ...
           'Shuffle', 1);
  h5write(image_fn, '/main', im);

end
