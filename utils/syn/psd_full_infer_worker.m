function psd_full_infer_worker(psd_model_fn, work_dir, num_cubes)

  psdm = load(psd_model_fn);

  if(isempty(gcp('nocreate')))
    tmp_cl = parcluster('local');
    parpool('local',tmp_cl.NumWorkers);
  end

  while(true)
    % check for status/*.ready
    dd = dir(sprintf('%s/status/*.ready', work_dir));
    % if empty, pause, continue
    if(isempty(dd))
      % check for status/*.done, break if length==num_cubes
      dd = dir(sprintf('%s/status/*.done', work_dir));
      if(length(dd) == num_cubes)
        break
      else
        pause(5)
        continue
      end
    end

    % run infer on oldest images/fn.h5
    [~,ii]      = min([dd.datenum]);
    [~,base_fn] = fileparts(dd(ii).name);
    seg_fn      = sprintf('%s/%s_seg.h5',     work_dir, base_fn);
    image_fn    = sprintf('%s/%s_image.h5',   work_dir, base_fn);
    json_pd     = sprintf('%s/%s_tbars.json', work_dir, base_fn);
    vars_fn     = sprintf('%s/%s_vars.mat',   work_dir, base_fn);
    fn_mat      = sprintf('%s/%s_syn.mat',    work_dir, base_fn);

    vars = load(vars_fn);
    % generate features
    [~, ff, ~, locs] = synapse_labels_feats(...
        json_pd, seg_fn, image_fn, [], [], ...
        vars.vol_sz_outer, ...
        psdm.window_radii, psdm.dilate_radii, psdm.image_thresh, ...
        [], true, ... % use_v2
        [], psdm.total_window, psdm.neighboring_dilate, ...
        true, 0, true);

    if(~isempty(ff))
      % feats2ind, normalize features, do inference
      if(~isempty(psdm.ff_thresh))
        ff_ind = feats2ind(ff, psdm.ff_thresh);
        ff     = [ff ff_ind];
      end
      ff = bsxfun(@rdivide, bsxfun(@minus, ff, psdm.fmn), ...
                  psdm.fstd+1e-4);

      % do classification with saved model
      pp = (psdm.psd_model.do_inference(ff) + 1) / 2; %#ok<PFBNS>

      % make sure to have both T-bar and PSD confidences
      [tlocs, plocs] = synapse_write_json(...
          [], locs, pp, [], psdm.psd_thresh, vars.tbars_local);

      % add in any T-bars with no PSDs
      [~,tm_idx] = setdiff(vars.tbars_local(1:3,:)', ...
                           tlocs(1:3,:)', 'rows');
      if(~isempty(tm_idx))
        n_empty = length(tm_idx);
        tlocs = [tlocs vars.tbars_local(:, tm_idx)];
        plocs(end+1:end+n_empty) = ...
            cell(1,n_empty);
      end
    else
      tlocs = vars.tbars_local;
      plocs = cell(1, size(tlocs,2));
    end

    % convert to global DVID coordinates
    tlocs(2,:)   = vars.vol_sz_outer(2) - tlocs(2,:) - 1;
    tlocs(1:3,:) = bsxfun(@plus, tlocs(1:3,:), ...
                          vars.vol_start_outer');
    for jj=1:size(tlocs,2)
      if(~isempty(plocs{jj}))
        plocs{jj}(2,:)   = vars.vol_sz_outer(2) - plocs{jj}(2,:) - 1;
        plocs{jj}(1:3,:) = bsxfun(@plus, plocs{jj}(1:3,:), ...
                                  vars.vol_start_outer');
      end
    end

    % save in .mat files for fast merging
    save(fn_mat, 'tlocs', 'plocs');

    tlocs_all{ii} = tlocs;
    plocs_all{ii} = plocs;

    % touch status/fn.infer, rm status/fn.image
    system(sprintf('touch %s/status/%s.done', work_dir, base_fn));
    system(sprintf('rm %s/status/%s.ready',   work_dir, base_fn));

    system(sprintf('rm %s', json_pd));
    system(sprintf('rm %s', vars_fn));
  end

end
