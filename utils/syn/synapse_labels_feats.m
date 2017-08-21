function [ll, ff, syn, locs] = synapse_labels_feats(...
  synapse_json, ...
  seg_fn, image_fn, tbar_fn, ...
  remove_buffer_radius, vol_sz, ...
  window_radii, dilate_radii, ...
  image_thresh, tbar_thresh, ...
  use_v2, pooling_type, ...
  total_window, neighboring_dilate, ...
  all_seg_candidates, offset, do_inference)
% SYNAPSE_LABELS_FEATS get labels and features (for libsvm)
% [ll, ff] = SYNAPSE_LABELS_FEATS(...
%   synapse_json, seg_fn, image_fn, tbar_fn, ...
%   remove_buffer_radius, vol_sz, ...
%   window_radii, dilate_radii, image_thresh, tbar_thresh, ...
%   dawmr_obj, ...
%   total_window, neighboring_dilate)
%
%   total_window defaults to 40
%   neighboring_dilate defaults to 4
%     (use -1 for no neighboring constraint)

  global DFEVAL_DIR

  if(~exist('remove_buffer_radius', 'var') || ...
     isempty(remove_buffer_radius))
    remove_buffer_radius = [];
    vol_sz               = [];
  else
    assert(...
      exist('vol_sz','var')>0 && ~isempty(vol_sz), ...
      'vol_sz must be specified if remove_buffer_radius used');
  end

  if(~exist('use_v2','var') || isempty(use_v2))
    use_v2 = false;
  end

  if(~exist('pooling_type','var') || isempty(pooling_type))
    pooling_type = 0;
  end
  if(~exist('total_window','var') || isempty(total_window))
    total_window = 40;
  end
  if(~exist('neighboring_dilate','var') || ...
     isempty(neighboring_dilate))
    neighboring_dilate = 4;
  end
  if(~exist('all_seg_candidates','var') || ...
     isempty(all_seg_candidates))
    all_seg_candidates = false;
  end
  if(~exist('offset','var'))
    offset = [];
  end
  if(~exist('do_inference','var') || isempty(do_inference))
    do_inference = false;
  end

  [total_window_flt, total_window_dd] = ...
      fml_set_filter(total_window);
  if(neighboring_dilate > 0)
    neighboring_flt = fml_set_filter(neighboring_dilate);
  else
    neighboring_flt = [];
  end

  %% read in ground-truth synapses
  [tbars, psds] = tbar_json2locs(synapse_json, offset);
  if(~isempty(remove_buffer_radius))
    if(do_inference)
      tbars = tbar_remove_border(...
        tbars, vol_sz, remove_buffer_radius);
      psds = psds(1,1:size(tbars,2));
    else
      [tbars, psds] = tbar_psd_remove_border(...
        tbars, psds, vol_sz, remove_buffer_radius);
    end
  end

  [psds_seg, seg_max] = tbar_locs2seg([{tbars} psds], seg_fn);
  all_segs   = unique(cell2mat(psds_seg));
  all_segs(all_segs==0) = [];
  tbars_seg  = psds_seg{1};
  psds_seg   = psds_seg(2:end);
  if(all_seg_candidates)
    valid_segs = []; %ones(1,seg_max);
  else
    valid_segs = zeros(1,seg_max);
    valid_segs(all_segs) = 1;
  end

  %% prepare data for distributed computation
  tbars_cell     = mat2cell(tbars, 3, ones(1, size(tbars,2)));
  tbars_seg_cell = num2cell(tbars_seg);

  zero_idx = tbars_seg == 0;
  tbars_cell(    zero_idx) = [];
  tbars_seg_cell(zero_idx) = [];
  psds_seg(      zero_idx) = [];

  dawmr_obj_fn = [];
  % if(~isempty(dawmr_obj))
  %   while(true)
  %     dawmr_obj_fn = sprintf('%s/dawmr_obj_%s_%s.mat', DFEVAL_DIR, ...
  %                            datestr(now,30), get_random_id([],1));
  %     if(exist(dawmr_obj_fn,'file')==0)
  %       break
  %     end
  %     pause(10);
  %   end
  %   save(dawmr_obj_fn, 'dawmr_obj');
  % end

  % fml_exe = fml_get_exe(true);
  % qret = fml_qsub(...
  %   @synapse_labels_feats_worker, ...
  %   [],[],[],-1,DFEVAL_DIR, fml_exe, ...
  %   tbars_cell, tbars_seg_cell, psds_seg, ...
  %   {valid_segs}, {seg_fn}, {image_fn}, {tbar_fn}, ...
  %   {neighboring_flt}, {total_window_flt}, ...
  %   {total_window}, {total_window_dd}, ...
  %   {window_radii}, {dilate_radii}, ...
  %   {image_thresh}, {tbar_thresh}, ...
  %   {dawmr_obj_fn}, {pooling_type});

  % n_qret = length(qret);
  % ll     = cell(n_qret,1);
  % ff     = cell(n_qret,1);
  % syn    = cell(n_qret,1);
  % locs   = cell(n_qret,1);
  % for ii=1:n_qret
  %   ll{ii}   = qret{ii}{1};
  %   ff{ii}   = qret{ii}{2};
  %   syn{ii}  = qret{ii}{3};
  %   locs{ii} = qret{ii}{4};
  % end

  n_syn = length(tbars_cell);
  ll    = cell(n_syn,1);
  ff    = cell(n_syn,1);
  syn   = cell(n_syn,1);
  locs  = cell(n_syn,1);

  if(use_v2)
    parfor ii = 1:n_syn
      [ll{ii}, ff{ii}, syn{ii}, locs{ii}] = ...
          synapse_labels_feats_worker2(...
              tbars_cell{ii}, tbars_seg_cell{ii}, psds_seg{ii}, ...
              valid_segs, seg_fn, image_fn, tbar_fn, ...
              neighboring_flt, total_window_flt, ...
              total_window, total_window_dd, ...
              window_radii, dilate_radii, ...
              image_thresh, tbar_thresh, ...
              dawmr_obj_fn, pooling_type);
    end
  else
    parfor ii = 1:n_syn
      [ll{ii}, ff{ii}, syn{ii}, locs{ii}] = ...
          synapse_labels_feats_worker(...
              tbars_cell{ii}, tbars_seg_cell{ii}, psds_seg{ii}, ...
              valid_segs, seg_fn, image_fn, tbar_fn, ...
              neighboring_flt, total_window_flt, ...
              total_window, total_window_dd, ...
              window_radii, dilate_radii, ...
              image_thresh, tbar_thresh, ...
              dawmr_obj_fn, pooling_type);
    end
  end

  ll   = cell2mat(ll);
  ff   = cell2mat(ff);
  syn  = cell2mat(syn);
  locs = cell2mat(locs);

  % if(~isempty(dawmr_obj_fn))
  %   system(sprintf('rm %s', dawmr_obj_fn));
  % end
end
