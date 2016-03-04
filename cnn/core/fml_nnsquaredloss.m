function Y = fml_nnsquaredloss(X,c,dzdy)
% FML_NNSQUAREDLOSS CNN squared loss
%   normalize by output size (ignores batch_size)
% Y = FML_NNSQUAREDLOSS(X,c,dzdy)

  s_sz = size(X);
  % normalize for everything except batch_size
  if(length(s_sz)<=3)
    s_vl = prod(s_sz);
  else
    s_vl = prod(s_sz(1:end-1));
  end

  if(nargin <= 2)
    t = (c - X).^2;
    Y = 0.5 * sum(t(:)) / s_vl;
  else
    Y = (X - c) .* dzdy / s_vl;
  end
end
