function Y = fml_nnsigentloss(X,c,target,dzdy)
% FML_NNSIGENTLOSS CNN combined sigmoid and cross-entropy loss
%   normalize by output size (ignores batch_size)
% Y = FML_NNSIGENTLOSS(X,c,target,dzdy)

  s = 1 ./ (1 + exp(-X));
  
  s_sz = size(s);
  % normalize for everything except batch_size
  if(length(s_sz)<=3)
    s_vl = prod(s_sz);
  else
    s_vl = prod(s_sz(1:end-1));
  end
  
  if(nargin <= 3)  
    tol = 1e-5;
    s = min(s,1-tol);
    s = max(s,  tol);
    t = -(c== 1) .* ...
        (target .* log(s) + (1-target) .* log(1-s)) ...
        -(c==-1) .* ...
        ((1-target) .* log(s) + target .* log(1-s));    
    Y = sum(t(:)) / s_vl;
  else
    Y = (c== 1) .* (s-target) + ...
        (c==-1) .* (s-(1-target));
    Y = Y .* dzdy / s_vl;
  end
end
