function [tloc, ploc] = tbar_psd_remove_border(...
  tloc, ploc, sz, buffer)

  idx = (tloc(1,:)<=buffer);
  tloc(:, idx) = [];
  ploc(idx)    = [];

  idx = (tloc(1,:)>sz(1) - buffer);
  tloc(:, idx) = [];
  ploc(idx)    = [];

  idx = (tloc(2,:)<=buffer);
  tloc(:, idx) = [];
  ploc(idx)    = [];

  idx = (tloc(2,:)>sz(2) - buffer);
  tloc(:, idx) = [];
  ploc(idx)    = [];

  idx = (tloc(3,:)<=buffer);
  tloc(:, idx) = [];
  ploc(idx)    = [];

  idx = (tloc(3,:)>sz(3) - buffer);
  tloc(:, idx) = [];
  ploc(idx)    = [];

  nn = length(ploc);

  psd_empty = false(1, nn);
  for ii=1:nn
    ploc{ii} = tbar_remove_border(ploc{ii}, sz, buffer);
    if(isempty(ploc{ii}))
      psd_empty(ii) = true;
    end
  end

  tloc(:, psd_empty) = [];
  ploc(psd_empty)    = [];
end
