function [lltt, fftt, syntt, locstt] = synapse_labels_feats_worker(...
  tbar, tbar_seg, psd_seg, valid_segs, ...
  seg_fn, image_fn, tbar_fn, ...
  neighboring_flt, total_window_flt, ...
  total_window, total_window_dd, ...
  window_radii, dilate_radii, image_thresh, tbar_thresh, ...
  dawmr_obj_fn, pooling_type)

  % establish ground-truth, remove 0s, self, and duplicates
  ii_psds_seg = psd_seg(psd_seg>0 & psd_seg~=tbar_seg);
  ii_psds_seg = unique(ii_psds_seg);

  hh  = h5info(seg_fn);
  hh_file = sprintf('/%s', hh.Datasets.Name);
  seg_sz1 = get_h5_size(seg_fn,hh_file);
  seg_sz1 = seg_sz1(1);

  tx = seg_sz1 - tbar(2);
  ty = tbar(1) + 1;
  tz = tbar(3) + 1;

  vol_sz = get_h5_size(image_fn);

  local_vol = double(read_image_stack_wpad(...
    seg_fn, hh_file, ...
    [tx ty tz] - total_window, ...
    (2*total_window+1)*ones(1,3)));

  % establish candidates
  ii_psds_cand = get_candidate_psds( ...
    tbar_seg, local_vol, neighboring_flt, total_window_flt);

  if(size(window_radii,1) == 1)
    window_radii(2,:) = NaN;
  end

  % window_radii = sort(window_radii);
  n_windows    = size(window_radii,2);
  n_dilate     = length(dilate_radii);
  n_image_t    = length(image_thresh);
  n_tbar_t     = length(tbar_thresh);
  n_dawmr      = 0;

  dawmr_obj = [];
  % if(~isempty(dawmr_obj_fn))
  %   dawmr_obj = load(dawmr_obj_fn, 'dawmr_obj');
  %   dawmr_obj = dawmr_obj.dawmr_obj;

  %   n_dawmr = length(pooling_type);
  %   dawmr_obj.ds.data_fn = { image_fn };
  % end
  n_features   = n_windows * n_dilate * ...
      (1 + n_image_t + n_tbar_t + n_image_t * n_tbar_t + ...
       n_dawmr);

  window_flt = cell(1, n_windows);
  for ii=1:n_windows
    window_flt{ii} = fml_set_filter(window_radii(1,ii));
  end
  dilate_flt = cell(1, n_dilate);
  for ii=1:n_dilate
    dilate_flt{ii} = fml_set_filter(dilate_radii(ii));
  end

  % first create common data
  window_offset = cell(n_windows,1);
  local_seg     = cell(n_windows,1);
  local_image   = cell(n_windows,1);
  local_tbar    = cell(n_windows,1);
  ts_mask       = cell(n_windows,n_dilate);

  for ww = 1:n_windows
    wwr = window_radii(1,ww);
    offset = [tx ty tz] - wwr;
    imdim  = (2*wwr+1)*ones(1,3);
    window_offset{ww} = offset - 1;

    local_seg{ww}   = double(read_image_stack_wpad(...
      seg_fn, hh_file, offset, imdim));

    local_seg{ww}   = local_seg{ww} .* window_flt{ww};
    local_image{ww} = read_image_stack_wpad(...
      image_fn, [],     offset, imdim);

    if(~isnan(window_radii(2,ww))) % ignore segm where image < thresh
      ignore_seg = local_image{ww} < window_radii(2,ww);
      local_seg{ww}(ignore_seg) = 0;
    end

    if(~isempty(tbar_fn))
      local_tbar{ww}  = read_image_stack_wpad(...
        tbar_fn,  [],     offset, imdim);
    end
    for dd = 1:n_dilate
      ts_mask{ww,dd} = double(convolution3D_FFTdomain( ...
        double(local_seg{ww} == tbar_seg), ...
        dilate_flt{dd}) > 1e-4);
    end
  end

  % set labels, features
  ii_psds_cand = ii_psds_cand(:)'; % is sorted from unique
  if(~isempty(valid_segs))
      ii_psds_cand(~valid_segs(ii_psds_cand)) = []; % remove invalid
  end
  n_cand  = length(ii_psds_cand);
  lltt    = zeros(n_cand,1);
  fftt    = zeros(n_cand,n_features);
  syntt   = zeros(n_cand,2);
  locstt  = zeros(n_cand,6);
  for ssi = 1:n_cand
    ss = ii_psds_cand(ssi);
    lltt(ssi,1) = ...
        2 * ~isempty(find(ii_psds_seg == ss, 1)) - 1;
    fftt(ssi,:) = ...
        get_features(ss, n_features, n_windows, n_dilate, ...
                         local_seg, local_image, local_tbar, ...
                         image_thresh, tbar_thresh, ...
                         dilate_flt, ts_mask, ...
                         dawmr_obj, window_offset, vol_sz, ...
                         pooling_type);
    syntt(ssi,:) = [tbar_seg, ss];

    locstt(ssi,:) = [...
      tbar', ...
      get_closest_point(...
        ss, local_vol, ...
        total_window, total_window_dd, tbar')];
  end

end

function cc = get_candidate_psds(ts, local_vol, ...
                                     neighboring_flt, ...
                                     total_window_flt)

  if(~isempty(neighboring_flt))
    % enforce neighboring constraint
    neighboring = double(convolution3D_FFTdomain(...
        double(local_vol == ts), ...
        neighboring_flt) > 1e-4);

    local_vol = local_vol .* neighboring;
  end

  local_vol = local_vol .* total_window_flt;

  cc = unique(local_vol(:));
  cc(cc == 0 | cc == ts) = [];
end

function feats = get_features(...
  ss, n_features, n_windows, n_dilate, ...
  local_seg, local_image, local_tbar, ...
  image_thresh, tbar_thresh, ...
  dilate_flt, ts_mask, ...
  dawmr_obj, window_offset, vol_sz, pooling_type)

  feats = zeros(1,n_features);
  idx   = 0;

  for wwc = 1:n_windows
    for ddc = 1:n_dilate
      ss_mask = double(convolution3D_FFTdomain(...
        double(local_seg{wwc} == ss), ...
        dilate_flt{ddc}) > 1e-4);

      in_mask = double(ts_mask{wwc,ddc} & ss_mask);

      idx = idx + 1;
      feats(idx) = sum(in_mask(:));

      for im_t = image_thresh(:)'
        im_mask = double(local_image{wwc} < im_t) .* in_mask;
        idx = idx + 1;
        feats(idx) = sum(im_mask(:));
      end

      for tb_t = tbar_thresh(:)'
        tb_mask = double(local_tbar{wwc} > tb_t) .* in_mask;
        idx = idx + 1;
        feats(idx) = sum(tb_mask(:));
      end

      for im_t = image_thresh(:)'
        for tb_t = tbar_thresh(:)'
          cb_mask = ...
              double(local_image{wwc} > im_t) .* ...
              double(local_tbar{wwc}  > tb_t) .* in_mask;
          idx = idx + 1;
          feats(idx) = sum(cb_mask(:));
        end
      end

      % TODO: clean this code up
      % if(~isempty(dawmr_obj))

      %   inds = find(in_mask>0);
      %   if(~isempty(inds))
      %     [xx,yy,zz] = ind2sub(size(in_mask), inds);
      %     xx = xx + window_offset{wwc}(1);
      %     yy = yy + window_offset{wwc}(2);
      %     zz = zz + window_offset{wwc}(3);

      %     inds = sub2ind(vol_sz, xx, yy, zz);
      %     for ii=1:length(dawmr_obj.dds)
      %       dd = dawmr_obj.dds(ii);
      %       res = dd.scaling;
      %       if(dd.downsampling==1)
      %         res = [1 1 1];
      %       end
      %       inds = dd.filter_indices(inds, ceil(vol_sz./res));
      %     end
      %     [xx,yy,zz] = ind2sub(vol_sz, inds);

      %     dawmr_feats = dawmr_obj.get_features(...
      %       inds,xx,yy,zz, 1, 1, []);

      %     switch(dawmr_obj.svm_normalization)
      %       case 0
      %       case 1
      %         assert(~isempty(dawmr_obj.sn1_max{1}), ...
      %                'DAWMRLIB:AssertionFailed', ...
      %                'normalization parameter not set');
      %         dawmr_feats = ...
      %             bsxfun(@rdivide, dawmr_feats, dawmr_obj.sn1_max{1});
      %       case 2
      %         assert(~isempty(dawmr_obj.sn2_min{1}) && ...
      %                ~isempty(dawmr_obj.sn2_max{1}), ...
      %                'DAWMRLIB:AssertionFailed', ...
      %                'normalization parameter not set');
      %         dawmr_feats = 2*bsxfun(@rdivide, ...
      %                                bsxfun(@minus, dawmr_feats, ...
      %                                       dawmr_obj.sn2_min{1}), ...
      %                                dawmr_obj.sn2_max{1}) - 1;
      %       case 3
      %         assert(~isempty(dawmr_obj.sn3_mn{1}) && ...
      %                ~isempty(dawmr_obj.sn3_std{1}), ...
      %                'DAWMRLIB:AssertionFailed', ...
      %                'normalization parameter not set');
      %         dawmr_feats = bsxfun(@rdivide, ...
      %                              bsxfun(@minus, dawmr_feats, ...
      %                                     dawmr_obj.sn3_mn{1}), ...
      %                              dawmr_obj.sn3_std{1});
      %       otherwise
      %         assert(0, 'DAWMRLIB:AssertionFailed', ...
      %                'unknown svm normalization');
      %     end

      %     vals_pd = ...
      %         dawmr_obj.end_classifier.do_inference(dawmr_feats);

      %     for thresh = pooling_type(:)'
      %       idx = idx + 1;
      %       feats(idx) = sum( max(0, vals_pd - thresh) );
      %     end

      %   end
      % end

    end
  end
end

function new_loc = get_closest_point(...
  ss, local_vol, total_window, total_window_dd, orig_loc)

  mm = total_window_dd;

  not_valid_mask = double(local_vol ~= ss);
  erode_r        = 3;
  erode_flt      = fml_set_filter(erode_r);
  not_valid_mask = logical(convolution3D_FFTdomain(...
      not_valid_mask, erode_flt) > 1e-4);
  mm(not_valid_mask) = Inf;

  [vv,idx]    = min(mm(:));

  if(vv == Inf)
      mm = total_window_dd;

      not_valid_mask = logical(local_vol ~= ss);
      mm(not_valid_mask) = Inf;

      [~,idx]    = min(mm(:));
  end

  [sx,sy,sz] = ind2sub(size(mm), idx);

  sx = sx - total_window - 1;
  sy = sy - total_window - 1;
  sz = sz - total_window - 1;

  new_loc = orig_loc + [sy, -sx, sz];
end
