function d = get_h5_size(fn, loc)
% GET_H5_SIZE get size of matrix stored in hdf5 file
% d = GET_H5_SIZE(fn, loc) loc defaults to '/main'
  
  if(~exist('loc','var') || isempty(loc))
    loc = '/main';
  end
  
  assert(exist(fn, 'file') > 0, 'JANCOM:AssertionFailed', ...
         sprintf('file %s does not exist', fn));
  
  h = h5info(fn, loc);
  d = h.Dataspace.Size;
end
