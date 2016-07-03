function [auc_roc, tpr, fpr, ...
          auc_pr, prec, recall, ...
          thds] = ...
    compute_auc(labels_gt, labels_pd, plot_title)

  if(~exist('plot_title','var'))
    plot_title = [];
  end
  if(~isempty(plot_title))
    assert(isstr(plot_title), 'plot_title should be string');
  end
  
  if(~iscell(labels_gt))
    labels_gt = {labels_gt};
  end
  if(~iscell(labels_pd))
    labels_pd = {labels_pd};
  end
  
  for i=1:length(labels_gt)
    n  = length(labels_gt{i});
    np = sum(labels_gt{i}>0);
    nn = n-np;
    
    tp = np;
    fp = nn;
    tn = 0;
    fn = 0;
    
    [~,ind] = sort(labels_pd{i});
    tmp_ord = labels_gt{i}(ind);
    
    fn = cumsum(tmp_ord > 0);
    tp = np - fn;
    tn = cumsum(tmp_ord <= 0);
    fp = nn - tn;
    
    prec{i}   = [np/(np+nn); tp./(tp+fp)];
    recall{i} = [1         ; tp./(tp+fn)];
    prec{i}(isnan(prec{i})) = 0;
    
    auc_pr(i) = .5*sum( (recall{i}(1:n)-recall{i}(2:(n+1))) .* ...
                        (prec{i}(1:n)+prec{i}(2:(n+1))) );
    
    fpr{i} = fp/nn;
    tpr{i} = tp/np;
    auc_roc(i) = .5*sum( (fpr{i}(1:(n-1))-fpr{i}(2:n)) .* ...
                         (tpr{i}(1:(n-1))+tpr{i}(2:n)) );
    thds{i} = [-Inf; labels_pd{i}(ind)];
    % slow sequential way:
    
    % prec{i}(1)   = tp/(tp+fp);
    % recall{i}(1) = 1;
    % auc{i}       = 0;

    % for j=1:n
    %   if(tmp_ord(j) > 0)
    %     fn = fn + 1;
    %     tp = tp - 1;
    %   else
    %     fp = fp - 1;
    %     tn = tn + 1;
    %   end
    %   prec{i}(j+1)   = tp/(tp+fp);
    %   if(isnan(prec{i}(j+1)))
    %     prec{i}(j+1) = prec{i}(j);
    %   end
    %   recall{i}(j+1) = tp/(tp+fn);
      
    %   auc{i} = auc{i} + ...
    %            .5 * (recall{i}(j)-recall{i}(j+1)) * ...
    %            (prec{i}(j+1)+prec{i}(j));
    % end
  end
  
  if(isstr(plot_title))
    plot_roc(tpr,fpr,auc_roc,plot_title);
  end
end
