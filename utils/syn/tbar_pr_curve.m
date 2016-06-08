function [pps, rrs, num_tp, tot_pred, tot_gt, ...
          num_tp_cont, tp_scores] = tbar_pr_curve(...
              fn_predict, fn_groundtruth, dist_thresh, thds, ...
              remove_buffer_radius, vol_sz, allow_mult, ...
              offset_gt, offset_pd, seg_fn, require_full_conf, ...
              flip_y)
% TBAR_PR_CURVE compute tbar precision/recall
% [pps, rrs, num_tp, tot_pred, tot_gt, ...
%  num_tp_cont, tp_scores] = TBAR_PR_CURVE(...
%        fn_predict, fn_groundtruth, dist_thresh, thds, ...
%        remove_buffer_radius, vol_sz, allow_mult, ...
%        offset_gt, offset_pd, seg_fn, require_full_conf)
%
%   fn_predict             prediction json filename
%   fn_groundtruth         groundtruth json filename
%   dist_thresh            max dist for a match
%   thds                   confidence thresholds to apply
%
% optional:
%   remove_buffer_radius   size to ignore on sides of volume
%   vol_sz                 volume size
%   allow_mult             allow pred to match multiple gt
%   offset_gt              shift groundtruth locations
%   offset_pd              shift predict locations
%                            (defaults to offset_gt)
%   seg_fn                 segmentation filename
%                            (to apply segmentation constraint)
%   require_full_conf      only use gt loc with conf=1.0
%   flip_y                 set to true if files are in differing
%                            coordinates (raveler/dvid)


  if(~exist('remove_buffer_radius', 'var') || ...
     isempty(remove_buffer_radius))
    remove_buffer_radius = [];
  else
    assert(...
      exist('vol_sz','var')>0 && ~isempty(vol_sz), ...
      'FML:AssertionFailed', ...
      'vol_sz must be specified if remove_buffer_radius used');
  end
  if(~exist('allow_mult','var'))
    allow_mult = [];
  end
  if(~exist('offset_gt','var'))
    offset_gt = [];
  end
  if(~exist('offset_pd','var'))
    offset_pd = offset_gt;
  end
  if(~exist('seg_fn','var'))
    seg_fn = [];
  end
  if(~exist('require_full_conf','var') || ...
     isempty(require_full_conf))
    require_full_conf = false;
  end
  if(~exist('flip_y','var') || isempty(flip_y))
    flip_y = false;
  end
  assert(~flip_y || ~isempty(vol_sz), ...
         'FML:AssertionFailed', ...
         'must specify vol_sz if using flip_y');

  locs_groundtruth = tbar_json2locs(fn_groundtruth, offset_gt, ...
				    true);
  if(require_full_conf)
    conf_idx = locs_groundtruth(4,:) > 0.99;
    locs_groundtruth = locs_groundtruth(1:3, conf_idx);
  else
    locs_groundtruth = locs_groundtruth(1:3,:);
  end
  if(~isempty(remove_buffer_radius))
    locs_groundtruth = tbar_remove_border(...
      locs_groundtruth, vol_sz, remove_buffer_radius);
  end
  if(~isempty(seg_fn))
    seg_groundtruth = tbar_locs2seg(locs_groundtruth, seg_fn);
  else
    seg_groundtruth = [];
  end

  n_thds   = length(thds);
  pps      = zeros(n_thds,1);
  rrs      = zeros(n_thds,1);
  num_tp   = zeros(n_thds,1);
  tot_pred = zeros(n_thds,1);
  tot_gt   = zeros(n_thds,1);

  num_tp_cont = zeros(n_thds,1);
  tp_scores   = cell(n_thds,1);

  locs_predict_orig  = tbar_json2locs(fn_predict, offset_pd, ...
                                      true);
  if(~isempty(remove_buffer_radius))
    locs_predict_orig = tbar_remove_border(...
      locs_predict_orig, vol_sz, remove_buffer_radius);
  end
  if(flip_y)
    locs_predict_orig(2,:) = vol_sz(2) - ...
        locs_predict_orig(2,:) - 1;
  end

  for ii = 1:n_thds
    tt = thds(ii);

    valid_tbars = locs_predict_orig(4,:) >= tt;
    locs_predict = locs_predict_orig(1:3,valid_tbars);

    if(~isempty(seg_fn))
      seg_predict = tbar_locs2seg(locs_predict, seg_fn);
    else
      seg_predict = [];
    end

    % avoid issue when there is only a single or no predictions
    if(sum(valid_tbars) < 2)
      pps(ii)         = 0;
      rrs(ii)         = 0;
      num_tp(ii)      = 0;
      tot_pred(ii)    = 0;
      tot_gt(ii)      = tot_gt(ii-1);
      num_tp_cont(ii) = 0;
      tp_scores{ii}   = [];
    else
      [pps(ii), rrs(ii), num_tp(ii), tot_pred(ii), tot_gt(ii), ...
       num_tp_cont(ii), tp_scores{ii}] = ...
        tbar_pr(locs_predict, locs_groundtruth, dist_thresh, ...
                allow_mult, seg_predict, seg_groundtruth);
    end
  end

end
