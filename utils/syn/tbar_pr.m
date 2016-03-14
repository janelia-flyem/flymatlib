function [pp, rr, num_tp, tot_pred, tot_gt, ...
          num_tp_cont, tp_scores, mm] = tbar_pr(...
            locs_predict, locs_groundtruth, dist_thresh, ...
            allow_mult, ...
            seg_predict, seg_groundtruth)

% input: predicted locations, ground truth locations
% compute distances
% put into intlinprog format, solve
% use cutoff to determine precision recall values
  
  if(~exist('allow_mult','var'))
    allow_mult = [];
  end
  if(~exist('seg_predict','var'))
    seg_predict     = [];
    seg_groundtruth = [];
  end
  
  locs_groundtruth = permute(locs_groundtruth, [1 3 2]);
  dists = squeeze(sqrt(sum(...
    bsxfun(@minus, locs_predict, locs_groundtruth).^2)));
  
  if(~isempty(seg_predict))
    seg_mask = bsxfun(@ne, seg_predict', seg_groundtruth);
    seg_mask = (dist_thresh+1) * seg_mask .* ones(size(seg_mask));
    dists    = dists + seg_mask;
  end
  
  mm = tbar_match_locs(dists - dist_thresh, allow_mult);
  
  tp_scores   = mm .* (dist_thresh - dists)/dist_thresh;
  num_tp_cont = sum(tp_scores(:));
  tp_scores   = tp_scores(tp_scores>0);
  
  mm = mm .* (dists < dist_thresh);
  
  num_tp   = sum(mm(:));
  tot_pred = size(dists,1) + sum(max(sum(mm,2)-1,0));
  tot_gt   = size(dists,2);
  
  pp = num_tp / tot_pred;
  rr = num_tp / tot_gt;
  
end
