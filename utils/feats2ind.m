function [feats_new, feats_thresh] = feats2ind(...
  feats_old, feats_thresh)

  n_obs = size(feats_old,1);
  n_dim = size(feats_old,2);

  if(isscalar(feats_thresh))
    n_thresh        = feats_thresh;
    feats_thresh    = zeros(n_dim, n_thresh);
    do_set_thresh   = true;
    thresh_vals     = (1:(n_thresh))  ./(n_thresh+1);
    thresh_vals_woz = (1:(n_thresh-1))./(n_thresh);
  else
    n_thresh      = size(feats_thresh,2);
    do_set_thresh = false;
  end

  feats_new = zeros(n_obs, n_thresh, n_dim);

  for ii=1:n_dim
    if(do_set_thresh)
      tt = quantile(feats_old(:,ii), thresh_vals);
      if(tt(2)==0) % multiple threshold values at 0
        tt = quantile(feats_old(feats_old(:,ii)>0,ii), ...
                      thresh_vals_woz);
        tt = [0 tt]; %#ok<AGROW>
      end
      feats_thresh(ii,:) = tt;
    end

    feats_new(:,:,ii) = bsxfun(...
      @gt, repmat(feats_old(:,ii), [1 n_thresh]), ...
      feats_thresh(ii,:) );
  end

  feats_new = reshape(feats_new, [n_obs, n_thresh*n_dim]);
end
