function locs = tbar_shift_predictions_noseg(json_in, json_out, ...
                                             offset, locs, ...
                                             image_fn, max_shift)

  from_json = false;
  if(isempty(locs)) % need to read from json
    from_json = true;
    if(~exist('offset','var'))
      offset = [0 0 0];
    end
    locs   = tbar_json2locs(json_in, offset, true);
  end

  hh    = h5info(image_fn);
  im    = read_image_stack(image_fn);
  im_sz = size(im);

  flt_sz       = 2*max_shift+1;
  [xx, yy, zz] = ndgrid(...
    -max_shift:max_shift,-max_shift:max_shift,-max_shift:max_shift);
  dd  = reshape(sqrt(xx(:).^2 + yy(:).^2 + zz(:).^2), ...
                [flt_sz flt_sz flt_sz]);
  flt = double(dd <= max_shift);

  for jj=1:size(locs,2)
    if(from_json)
      my = locs(1,jj) + 1;
      mx = im_sz(1) - locs(2,jj);
      mz = locs(3,jj) + 1;
    else
      mx = locs(1,jj);
      my = locs(2,jj);
      mz = locs(3,jj);
    end

    if(mx - max_shift < 1 || mx + max_shift > im_sz(1) || ...
       my - max_shift < 1 || my + max_shift > im_sz(2) || ...
       mz - max_shift < 1 || mz + max_shift > im_sz(3))
      continue
    end

    local_vol  = double(im( mx-max_shift:mx+max_shift, ...
                            my-max_shift:my+max_shift, ...
                            mz-max_shift:mz+max_shift ));
    local_vol(~logical(flt)) = -Inf;

    [~,max_idx] = max(local_vol(:));
    [nx,ny,nz]  = ind2sub(size(local_vol), max_idx);

    mx = mx + nx - max_shift - 1;
    my = my + ny - max_shift - 1;
    mz = mz + nz - max_shift - 1;

    if(from_json)
      locs(1,jj) = my - 1 + offset(1);
      locs(2,jj) = im_sz(1) - mx;
      locs(3,jj) = mz - 1 + offset(3);
    else
      locs(1:3,jj) = [mx my mz]';
    end
  end

  if(from_json)
    tbar_json_write(json_out, locs);
  end
end
