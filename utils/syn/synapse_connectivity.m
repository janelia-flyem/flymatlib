function g = synapse_connectivity(labels_tt, labels_pp, ...
                                  tt, thresh_tt, ...
                                  pp, thresh_pp, ...
                                  allow_multiple_conn, ...
                                  allow_self)

  n_tt = length(labels_tt);
  if(exist('thresh_tt','var') && ~isempty(thresh_tt))
    apply_thresh_tt = true;
  else
    apply_thresh_tt = false;
  end
  if(exist('thresh_pp','var') && ~isempty(thresh_pp))
    apply_thresh_pp = true;
  else
    apply_thresh_pp = false;
  end
  if(~exist('allow_multiple_conn','var') || ...
     isempty(allow_multiple_conn))
    allow_multiple_conn = true;
  end
  if(~exist('allow_self','var') || ...
     isempty(allow_self))
    allow_self = true;
  end

  syn = cell(1,n_tt);

  for ii=1:n_tt
    if(labels_tt(ii) == 0)
      warning('FML:Warning', 't-bar with label 0');
      continue
    end
    if(apply_thresh_tt && tt(4,ii) < thresh_tt)
      continue
    end

    if(isempty(labels_pp{ii})), continue, end
    n_pp_z = sum(labels_pp{ii}==0);
    if(n_pp_z > 0)
      warning('FML:Warning', '%d psd with label 0', n_pp_z);
    end

    if(apply_thresh_pp)
      jj = find(labels_pp{ii} > 0 & ...
                pp{ii}(4,:) >= thresh_pp);
    else
      jj = find(labels_pp{ii} > 0);
    end

    labels_pp_sel = labels_pp{ii}(jj);
    if(~allow_multiple_conn)
      labels_pp_sel = unique(labels_pp_sel);
    end
    if(~allow_self)
      labels_pp_sel = setdiff(labels_pp_sel, labels_tt(ii));
    end
    n_pp = length(labels_pp_sel);
    if(n_pp == 0), continue, end

    syn{ii}      = zeros(2, n_pp);
    syn{ii}(1,:) = labels_tt(ii);
    syn{ii}(2,:) = labels_pp_sel;
  end

  syn  = cell2mat(syn);
  if(~isempty(syn))
    smax = max(syn(:));
    g    = sparse(syn(1,:),syn(2,:),1,smax,smax);
  else
    g    = sparse([]);
  end
end
