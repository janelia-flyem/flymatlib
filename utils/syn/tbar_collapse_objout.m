function [values_raw, values_conf, out] = tbar_collapse_objout(...
  image_fn, out_fn, obj_thresh, values_raw, values_conf)
% [values_raw, values_conf, out] = TBAR_COLLAPSE_OBJOUT(...
%     image_fn, out_fn, obj_thresh, values_raw, values_conf)
% collapse 4 channel object output (object + center predictions)
%   to single channel object prediction only
%
% image_fn     filename of 4 channel file
% out_fn       filename of 1 channel file to output
% obj_thresh   threshold on object voxels to consider (default 0.2)
% values_raw, values_conf
%              needed to convert raw values after collapsing to
%                single channel, which may be greater than 1,
%                into confidence values
%                if not specified, then set using empirical CDF
  
  if(ischar(image_fn))
    im = read_image_stack(image_fn);
  else
    im = image_fn;
  end
  if(~exist('obj_thresh','var') || isempty(obj_thresh))
    obj_thresh = 0.2;
  end
  
  sz = size(im);
  
  [xx,yy,zz] = ind2sub(sz(1:3), find(im(:,:,:,1)>obj_thresh));
  
  oo = ones(size(xx));
  xn = round(xx + im(sub2ind(sz,xx,yy,zz,2*oo)));
  yn = round(yy + im(sub2ind(sz,xx,yy,zz,3*oo)));
  zn = round(zz + im(sub2ind(sz,xx,yy,zz,4*oo)));

  out = zeros(sz(1:3),'single');
  for ii=1:length(xn)
    out(xn(ii),yn(ii),zn(ii)) = ...
        out(xn(ii),yn(ii),zn(ii)) + im(xx(ii),yy(ii),zz(ii),1);
  end

  if(~exist('values_raw','var') || isempty(values_raw))
    [values_conf, values_raw] = ecdf(out(out>0));
    % ensure values are unique for using in interp1
    [values_raw, values_idx] = unique(values_raw);
    values_conf              = values_conf(values_idx);
  end
  
  % convert raw values into confidences
  cc = interp1(values_raw, values_conf, out(out>0));
  out(out > 0) = cc;
  
  if(exist('out_fn','var') && ~isempty(out_fn))
    if(exist(out_fn,'file'))
      delete(out_fn);
    end
    chunk_size = [30 30 30];
    chunk_size = min(chunk_size, sz(1:3));
    h5create(out_fn,'/main',size(out),...
             'Datatype','single',...
             'Chunksize', chunk_size, ...
             'Deflate', 4, ...
             'Shuffle', 1);
    h5write(out_fn, '/main', out);
  end

end
