function Y = fml_nnl1loss(X,c,dzdy)
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

  if(nargin <= 2)
    t = abs(c - X);
    Y = sum(t(:)) / s_vl;
  else
    Y = ( (X>c) - (X<c) ) .* dzdy / s_vl;
  end
end
