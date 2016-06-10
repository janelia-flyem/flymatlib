function [ss, smax] = tbar_locs2seg(locs, seg_fn)

  if(ischar(seg_fn))
    hh  = h5info(seg_fn);
    hh_file = sprintf('/%s', hh.Datasets.Name);
    seg = hdf5read(seg_fn, hh_file);
  else
    seg = seg_fn;
  end

  not_cell = false;
  if(~iscell(locs))
    not_cell = true;
    locs = { locs };
  end

  nn = length(locs);
  ss = cell(1,nn);
  for ii=1:nn
    locs{ii}(2,:) = size(seg,1) - locs{ii}(2,:) - 1;
    locs{ii} = locs{ii}([2 1 3], :) + 1;

    idx = sub2ind(size(seg), ...
                  locs{ii}(1,:), locs{ii}(2,:), locs{ii}(3,:));

    ss{ii} = seg(idx);
  end

  if(not_cell)
    ss = ss{1};
  end

  if(nargout == 2)
    smax = max(seg(:));
  end
end
