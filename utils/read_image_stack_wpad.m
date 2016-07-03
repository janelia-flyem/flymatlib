function [image, offset_inner, imdim_inner, ...
          pad_front, pad_back] = ...
    read_image_stack_wpad(fn, p, offset, imdim)
% read hdf5 format image stack
% at offset and imdim

  if(~exist('p','var') || isempty(p)) % no path specified
    p = '/main'; % default to /main
  end

  sz = get_h5_size(fn,p);

  pad_front    = max(0, [1 1 1] - offset);
  offset_inner = offset + pad_front;

  last_idx       = offset + imdim - 1;
  pad_back       = max(0, last_idx - sz(1:3));
  last_idx_inner = last_idx - pad_back;
  imdim_inner    = last_idx_inner - offset_inner + 1;

  if(length(sz) > 3)
    offset_inner = [offset_inner 1];
    imdim_inner  = [imdim_inner  sz(4)];
  end

  image_inner = read_image_stack(fn, p, offset_inner, imdim_inner);

  image = zeros(imdim, class(image_inner));

  image(1+pad_front(1):end-pad_back(1), ...
        1+pad_front(2):end-pad_back(2), ...
        1+pad_front(3):end-pad_back(3)) = image_inner;

end
