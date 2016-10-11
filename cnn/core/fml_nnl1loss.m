function Y = fml_nnl1loss(X,c,l_eps,w,dzdy)
% FML_NNL1LOSS CNN L1 loss
%   normalize by output size (ignores batch_size)
% Y = FML_NNL1LOSS(X,c,dzdy)

  s_sz = size(X);
  % normalize for everything except batch_size
  if(length(s_sz)<=3)
    s_vl = prod(s_sz);
  else
    s_vl = prod(s_sz(1:end-1));
  end

  if(nargin <= 4)
    t = max(0, X-c) + max(0, c-X-l_eps);
    t = t .* (1+w*max(0,0-c));
    Y = sum(t(:)) / s_vl;
  else
    Y = ( (X>c) - (X<(c-l_eps)) ) .* ...
        (1+w*max(0,0-c)) .* dzdy / s_vl;
  end
end
