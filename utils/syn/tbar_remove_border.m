function locs = tbar_remove_border(locs, sz, buffer)
% locs = TBAR_REMOVE_BORDER(locs, sz, buffer)
%   remove tbars that are close to volume boundary
%
% locs     tbar locations
% sz       volume size
% buffer   size of buffer region in which to remove tbars
  
  locs(:, locs(1,:)<=buffer)        = [];
  locs(:, locs(1,:)>sz(1) - buffer) = [];
  locs(:, locs(2,:)<=buffer)        = [];
  locs(:, locs(2,:)>sz(2) - buffer) = [];
  locs(:, locs(3,:)<=buffer)        = [];
  locs(:, locs(3,:)>sz(3) - buffer) = [];
end
