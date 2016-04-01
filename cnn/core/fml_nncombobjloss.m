function Y = fml_nncombobjloss(X,c,target,robj,rnorm,dzdy)
% FML_NNCOMBOBJLOSS CNN combined object loss, expects
%   4 channel data where first channel is binary object-ness
%   prediction, and second through fourth channels are
%   predictions of x,y,z offsets to object center
% Y = FML_NNCOMBOBJLOSS(X,c,robj,rnorm,dzdy)
%   robj defines object center radius, as long as prediction
%   is within robj of ground-truth offset, no penalty is incurred
%   rnorm is used to divide offset predictions prior to
%   smoothed/robust L1 loss

  oc   = c(:,:,:,1,:);
  s_sz = size(X);
  % normalize for everything except batch_size
  s_vl = prod(s_sz(1:3));

  ff = (X(:,:,:,2:4,:) - c(:,:,:,2:4,:))/rnorm;
  ff = max(abs(ff)-robj/rnorm,0).*sign(ff);
  ff = bsxfun(@times, ff, oc>0);

  if(nargin <= 5)
    Y  = fml_nnsigentloss(X(:,:,:,1,:), oc, target);
    ff = (0.5*ff.^2).*(abs(ff)<1) + (abs(ff)-0.5).*(abs(ff)>=1);
    Y  = Y + sum(ff(:)) / s_vl;
  else
    Y  = fml_nnsigentloss(X(:,:,:,1,:), oc, target, dzdy);
    ff = (ff.*(abs(ff)<1) + sign(ff).*(abs(ff)>=1)) * ...
         dzdy / (rnorm * s_vl);
    Y  = cat(4, Y, ff);
  end
end
