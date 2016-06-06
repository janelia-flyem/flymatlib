function tbar_cnn_full_infer(work_dir, vol_start_fn, ...
                             vol_sz, buffer_sz, ...
                             do_normalize, ...
                             obj_thresh, ...
                             ave_radius, obj_min_dist, ...
                             dvid_conn, net_fn)

  fid = fopen(vol_start_fn);
  if(~isempty(vol_sz))
    vol_start = fscanf(fid, '%d', [3,Inf])';
    if(isscalar(vol_sz))
      vol_sz = vol_sz*ones(1,3);
    end
    vol_sz = repmat(vol_sz, size(vol_start,1),1);
  else
    vol_start = fscanf(fid, '%d', [6,Inf])';
    vol_sz    = vol_start(:,4:6);
    vol_start = vol_start(:,1:3);
  end
  fclose(fid);
  n_substacks = size(vol_start,1);

  system(sprintf('mkdir -p %s',        work_dir));
  system(sprintf('mkdir -p %s/status', work_dir));
  system(sprintf('mkdir -p %s/images', work_dir));
  system(sprintf('mkdir -p %s/infer', work_dir));
  system(sprintf('mkdir -p %s/json', work_dir));

  % shift over vol_start and expand vol_sz to accomodate buffer_sz
  if(isscalar(buffer_sz))
    buffer_sz = buffer_sz*ones(1,3);
  end
  vol_start = bsxfun(@minus, vol_start, buffer_sz);
  vol_sz    = bsxfun(@plus,  vol_sz,    2*buffer_sz);

  fprintf('tbar_cnn_full_infer_worker(''%s'', ''%s'', %d)\n', ...
          net_fn, work_dir, n_substacks);

  parfor ii=1:n_substacks
    base_fn = sprintf('%06d', ii);
    done_st = sprintf('%s/status/%s.done', work_dir, base_fn);
    % check in case job is from re-started state
    if(exist(done_st,'file')), continue, end

    % pull down image from dvid
    image_fn = sprintf('%s/images/%s.h5', work_dir, base_fn);
    dvid_conn.get_image(...
        vol_start(ii,:), vol_sz(ii,:), image_fn, do_normalize); %#ok<PFBNS>
    system(sprintf('touch %s/status/%s.image', work_dir, base_fn));

    infer_st = sprintf('%s/status/%s.infer', work_dir, base_fn);

    % wait for voxel-wise inference, then post-process
    while(~exist(infer_st, 'file')), pause(10), end
    infer_fn = sprintf('%s/infer/%s.h5',  work_dir, base_fn);
    json_fn  = sprintf('%s/json/%s.json', work_dir, base_fn);
    tbar_voxel2obj(infer_fn, json_fn, obj_thresh, ...
                   ave_radius, obj_min_dist);

    % touch status/fn.done, rm status/fn.infer
    system(sprintf('touch %s', done_st));
    system(sprintf('rm %s',    infer_st));
    % clean-up: remove image_fn, infer_fn
    system(sprintf('rm %s', image_fn));
    system(sprintf('rm %s', infer_fn));
  end
end
