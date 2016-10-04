function out = c3dae_infer(net, image_fn, out_fn, ...
                               corrupt_fraction, ...
                               corrupt_type)

  if(~exist('corrupt_fraction','var') || ...
     isempty(corrupt_fraction))
      corrupt_fraction = 0;
      corrupt_type     = -1;
  end

  tic
  if(ischar(image_fn))
    im = read_image_stack(image_fn);
  else
    im = image_fn;
  end
  toc

  tic
  im_sz   = size(im);
  im_prod = im_sz(1)*im_sz(2);

  out     = zeros(im_sz,'single');

  yz_step  = 80;

  net = fml_simplenn_move(net, 'gpu');
  for yy=1:yz_step:im_sz(2)
    y_end    = min(yy+yz_step-1,im_sz(2));
    for zz = 1:yz_step:im_sz(3)
      z_end    = min(zz+yz_step-1,im_sz(3));

      im_slice = gpuArray(im(:,yy:y_end,zz:z_end));
      if(corrupt_type == 0)
        rr = randsample(im_prod, ...
                        floor(numel(im_slice)*corrupt_fraction));
        im_slice(rr) = 0;
      end
      if(corrupt_type == 1)
        im_slice = im_slice + ...
            corrupt_fraction*randn(size(im_slice));
      end
      res    = fml_simplenn(net,im_slice,[],[],'mode','test');
      res_sz = size(res(end).x);
      out(1:res_sz(1),yy-1+(1:res_sz(2)),zz-1+(1:res_sz(3))) = ...
          gather(res(end).x);
    end
  end
  toc

  tic
  if(exist('out_fn','var') && ~isempty(out_fn))
    if(exist(out_fn,'file'))
      delete(out_fn);
    end
    chunk_size = [30 30 30];
    chunk_size = min(chunk_size, im_sz);
    h5create(out_fn,'/main',im_sz,...
             'Datatype','single',...
             'Chunksize', chunk_size, ...
             'Deflate', 4, ...
             'Shuffle', 1);
    h5write(out_fn, '/main', out);
  end
  toc
end
