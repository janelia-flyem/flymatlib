function image = read_image_stack(fn, p, offset, imdim)
% read hdf5 format image stack
% image = READ_IMAGE_STACK(fn, p, offset, imdim)
%   fn       filename
%   p        path, defaults to /main
%   offset   optional
%   imdim    optional
  
  if(~exist('p','var') || isempty(p)) % no path specified
    p = '/main'; % default to /main
  end
  
  if(exist('offset','var') && ~isempty(offset))
    if(~exist('imdim','var') || isempty(imdim))
      imdim = Inf * offset;
    end

    h  = h5info(fn,p);
    sz = length(h.Dataspace.Size);
    if(sz < length(offset))
      offset_extra = offset(sz+1:end);
      imdim_extra  = imdim( sz+1:end);
      assert( sum(offset_extra==1) == length(offset_extra), ...
              'FML:AssertionFailed', ...
              'dimension mismatch in read_image_stack' );
      assert( sum(imdim_extra==1) == length(imdim_extra), ...
              'FML:AssertionFailed', ...
              'dimension mismatch in read_image_stack' );

      offset = offset(1:sz);
      imdim  = imdim( 1:sz);
    end
    image = h5read(fn, p, offset, imdim);
  else
    image = h5read(fn, p);
  end
  
end
