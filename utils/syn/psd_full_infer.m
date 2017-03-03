function psd_full_infer(work_dir, psd_model_fn, tbars, ...
                        vol_start_fn, vol_sz, ...
                        dvid_conn, seg_name)

  psdm    = load(psd_model_fn);
  bg_vals = [0 255];

  fid = fopen(vol_start_fn);
  if(~isempty(vol_sz))
    vol_start = fscanf(fid, '%d', [3,Inf])';
    if(isscalar(vol_sz))
      vol_sz    = vol_sz*ones(1,3);
    end
    vol_sz    = repmat(vol_sz, size(vol_start,1),1);
  else
    vol_start = fscanf(fid, '%d', [6,Inf])';
    vol_sz    = vol_start(:,4:6);
    vol_start = vol_start(:,1:3);
  end
  fclose(fid);

  n_substacks = size(vol_start,1);

  system(sprintf('mkdir -p %s', work_dir));
  system(sprintf('mkdir -p %s/status', work_dir));

  fprintf('psd_full_infer_worker(''%s'', ''%s'', %d)\n', ...
          psd_model_fn, work_dir, n_substacks);

  % maximum buffer needed by PSD feature computation
  bf = max([psdm.window_radii(1,:) psdm.total_window]);

  % get T-bars per substack
  tbars_ss = cell(1,n_substacks);
  parfor ii=1:n_substacks
    idx = tbars(1,:) >= vol_start(ii,1) & ...
          tbars(1,:) <  vol_start(ii,1) + vol_sz(ii,1) & ...
          tbars(2,:) >= vol_start(ii,2) & ...
          tbars(2,:) <  vol_start(ii,2) + vol_sz(ii,2) & ...
          tbars(3,:) >= vol_start(ii,3) & ...
          tbars(3,:) <  vol_start(ii,3) + vol_sz(ii,3); %#ok<PFBNS>

    tbars_ss{ii} = tbars(:,idx);
  end

  tlocs_all = cell(1, n_substacks);
  plocs_all = cell(1, n_substacks);

  parfor ii=1:n_substacks
    % skip over any previously processed substacks
    fn_mat = sprintf('%s/%06d_synsh.mat', work_dir, ii);
    if(exist(fn_mat,'file')), continue, end

    % adding in buffer large enough to do PSD features
    vol_start_outer = vol_start(ii,:) - bf;
    vol_sz_outer    = vol_sz(ii,:)    + 2*bf;

    % get segmentation
    seg_fn = sprintf('%s/%06d_seg.h5', work_dir, ii);
    dvid_conn.get_segmentation(...
        vol_start_outer, vol_sz_outer, seg_fn, seg_name);
    % get image
    image_fn = sprintf('%s/%06d_image.h5', work_dir, ii);
    [~,~,empty_vol] = dvid_conn.get_image(...
        vol_start_outer, vol_sz_outer, image_fn, ...
        psdm.do_normalize, bg_vals);

    if(~empty_vol)
      % convert tbars to local Raveler coordinates
      tbars_local        = tbars_ss{ii};
      tbars_local(1:3,:) = bsxfun(@minus, tbars_local(1:3,:), ...
                                  vol_start_outer');
      tbars_local(2,:)   = vol_sz_outer(2) - tbars_local(2,:) - 1;

      json_pd = sprintf('%s/%06d_tbars.json', work_dir, ii);
      tbar_json_write(json_pd, tbars_local);

      vars_fn = sprintf('%s/%06d_vars.mat', work_dir, ii);
      save_vars(vars_fn, vol_start_outer, vol_sz_outer, tbars_local);

      ready_st = sprintf('%s/status/%06d.ready', work_dir, ii);
      done_st  = sprintf('%s/status/%06d.done',  work_dir, ii);
      system(sprintf('touch %s', ready_st));
      while(~exist(done_st, 'file')), pause(10), end

      fn_mat_orig = sprintf('%s/%06d_syn.mat', work_dir, ii);
      ss = load(fn_mat_orig, 'tlocs', 'plocs');
      tlocs = ss.tlocs;
      plocs = psd_shift_ann(tlocs, ss.plocs, dvid_conn, seg_name);
      plocs = psd_make_global_unique(plocs,  dvid_conn, seg_name);
      save_out(fn_mat, tlocs, plocs);
      system(sprintf('rm %s', fn_mat_orig));
    end

    system(sprintf('rm %s %s', seg_fn, image_fn));
  end

  % merge all tlocs, plocs, save in .mat
  for ii=1:n_substacks
    fn_mat = sprintf('%s/%06d_synsh.mat', work_dir, ii);
    ss = load(fn_mat, 'tlocs', 'plocs');
    tlocs_all{ii} = ss.tlocs;
    plocs_all{ii} = ss.plocs;
  end
  tlocs  = cell2mat(tlocs_all);
  plocs  = horzcat( plocs_all{:} );
  fn_syn = sprintf('%s/syn.mat', work_dir);
  save(fn_syn, 'tlocs', 'plocs');
end

function save_vars(vars_fn, vol_start_outer, vol_sz_outer, tbars_local)
  save(vars_fn, 'vol_start_outer', 'vol_sz_outer', 'tbars_local');
end

function save_out(fn, tlocs, plocs)
  save(fn, 'tlocs', 'plocs');
end
