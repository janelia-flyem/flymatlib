function [magnitude_pr, binary_pr] = ...
    synapse_pr_curve(gt_labels_tt, gt_labels_pp, ...
                     pd_labels_tt, pd_labels_pp, ...
                     pd_pp, pp_thds, filter_seg, ...
                     allow_multiple_conn, ...
                     allow_self)

  if(~exist('allow_multiple_conn','var'))
    allow_multiple_conn = [];
  end
  if(~exist('allow_self','var') || ...
     isempty(allow_self))
    allow_self = [];
  end

  gt = synapse_connectivity(gt_labels_tt, gt_labels_pp, ...
                            [],[],[],[], ...
                            allow_multiple_conn, allow_self);

  nn_gt = sum(gt,1)>0 | sum(gt,2)'>0;
  nn_gt = find(nn_gt);
  max_nn_gt = max(nn_gt);

  switch(filter_seg)
    case 'none'
      gt_sub = gt;
    case 'groundtruth'
      gt_sub = gt(nn_gt,nn_gt);
    case 'predicted'
    otherwise
      assert(false, 'FLYEMLIB:AssertionFailed', ...
	     sprintf('unknown filter_seg = %s', filter_seg));
  end

  n_thds                = length(pp_thds);
  magnitude_pr.pps      = zeros(n_thds,1);
  magnitude_pr.rrs      = zeros(n_thds,1);
  magnitude_pr.num_tp   = zeros(n_thds,1);
  magnitude_pr.tot_pred = zeros(n_thds,1);
  magnitude_pr.tot_gt   = zeros(n_thds,1);
  binary_pr.pps         = zeros(n_thds,1);
  binary_pr.rrs         = zeros(n_thds,1);
  binary_pr.num_tp      = zeros(n_thds,1);
  binary_pr.tot_pred    = zeros(n_thds,1);
  binary_pr.tot_gt      = zeros(n_thds,1);

  for ti = 1:n_thds
    gp = synapse_connectivity(pd_labels_tt, pd_labels_pp, ...
                              [],[], pd_pp, pp_thds(ti), ...
                              [], allow_self);

    if(size(gp,1) < max_nn_gt)
      gp(max_nn_gt, max_nn_gt) = 0;
    end

    switch(filter_seg)
      case 'none'
	gp_sub = gp;
      case 'groundtruth'
	gp_sub = gp(nn_gt,nn_gt);
      case 'predicted'
	nn     = nn_gt | sum(gp,2)';
        % nnc = sum(gp,2)' & ~nn_gt;
        % full(sum(sum(gp(nnc,nnc))))
	gt_sub = gt(nn,nn);
	gp_sub = gp(nn,nn);
    end

    if(size(gp_sub,1) < size(gt_sub,1))
      n_gt_sub = size(gt_sub,1);
      gp_sub(n_gt_sub,n_gt_sub) = 0;
    end
    if(size(gt_sub,1) < size(gp_sub,1))
      n_gp_sub = size(gp_sub,1);
      gt_sub(n_gp_sub,n_gp_sub) = 0;
    end

    % magnitude_pr
    mm = min(gt_sub,gp_sub);
    magnitude_pr.num_tp(ti)   = full(sum(sum(mm)));
    magnitude_pr.tot_pred(ti) = full(sum(sum(gp_sub)));
    magnitude_pr.tot_gt(ti)   = full(sum(sum(gt_sub)));

    % binary_pr
    mm = gt_sub & gp_sub;
    binary_pr.num_tp(ti)   = full(sum(sum(mm)));
    binary_pr.tot_pred(ti) = nnz(gp_sub);
    binary_pr.tot_gt(ti)   = nnz(gt_sub);
  end

  magnitude_pr.pps = magnitude_pr.num_tp ./ magnitude_pr.tot_pred;
  magnitude_pr.rrs = magnitude_pr.num_tp ./ magnitude_pr.tot_gt;

  binary_pr.pps = binary_pr.num_tp ./ binary_pr.tot_pred;
  binary_pr.rrs = binary_pr.num_tp ./ binary_pr.tot_gt;

  % NaN -> 0 / 0 -> convert to 1
  magnitude_pr.pps(isnan(magnitude_pr.pps)) = 1;
  magnitude_pr.rrs(isnan(magnitude_pr.rrs)) = 1;
  binary_pr.pps(isnan(binary_pr.pps)) = 1;
  binary_pr.rrs(isnan(binary_pr.rrs)) = 1;
end
