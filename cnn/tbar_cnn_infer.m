function out = tbar_cnn_infer(net, image_fn, out_fn, ...
                              n_workers, ...
                              offsets, copies)
% TBAR_CNN_INFER distributed tbar cnn inference
% out = TBAR_CNN_INFER(net, image_fn, out_fn, ...
%                      n_workers, offsets, copies)
%
%   net         matconvnet network
%   image_fn    image filename
%   out_fn      output filename
%   n_workers   number of distributed workers to allocate
%   offsets
%   copies

  global DFEVAL_DIR

  if(~exist('offsets','var') || isempty(offsets))
    offsets = zeros(2,1);
  end
  if(~exist('copies','var') || isempty(copies))
    copies  = cell(1,size(offsets,2));
  end

  im_sz    = get_h5_size(image_fn);
  net_info = fml_simplenn_display(...
    net, 'inputSize', [im_sz(1:2) 1 1]);

  rf_stride = net_info.receptiveFieldStride(:,end);
  rf_offset = net_info.receptiveFieldOffset(:,end);

  % assume rf stride/offset are all isotropic

  cube_sz = 70; % make this an optional parameter?
  [xx,yy,zz] = ndgrid( ...
    rf_offset(1):cube_sz:im_sz(1)-rf_offset(1)+1, ...
    rf_offset(1):cube_sz:im_sz(2)-rf_offset(1)+1, ...
    rf_offset(1):cube_sz:im_sz(3)-rf_offset(1)+1);

  xx=xx(:); yy=yy(:); zz=zz(:);

  last_conv = length(net.layers);
  while(~isfield(net.layers{last_conv},'weights'))
    last_conv = last_conv-1;
  end
  n_out = size(net.layers{last_conv}.weights{1},5);
  out   = zeros([im_sz n_out],'single');

  n_cubes = length(xx);
  coords  = zeros(n_cubes,6);

  for ii=1:n_cubes
    xs = xx(ii)-rf_offset(1)+1;
    xe = min(im_sz(1), xx(ii)+cube_sz+rf_offset(1)-2);
    ys = yy(ii)-rf_offset(1)+1;
    ye = min(im_sz(2), yy(ii)+cube_sz+rf_offset(1)-2);
    zs = zz(ii)-rf_offset(1)+1;
    ze = min(im_sz(3), zz(ii)+cube_sz+rf_offset(1)-2);

    coords(ii,:) = [xs ys zs (xe-xs+1) (ye-ys+1) (ze-zs+1)];
  end

  n_per_worker = floor(n_cubes / n_workers);
  n_with_extra = n_cubes - n_workers*n_per_worker;

  worker_loads = n_per_worker * ones(1,n_workers);
  worker_loads(1:n_with_extra) = n_per_worker+1;

  coords = mat2cell(coords, worker_loads, 6);

  while(true)
    net_fn = sprintf('%s/cnn_%s_%s.mat', DFEVAL_DIR, ...
                     datestr(now,30), get_random_id([],1));
    if(exist(net_fn,'file')==0), break, end
    pause(10);
  end
  save(net_fn, 'net');

  if(n_workers > 1)
    fml_exe = fml_get_exe(true);
    res = fml_qsub(@tbar_cnn_infer_worker, ...
                   [],[],[],-1, ...
                   DFEVAL_DIR, fml_exe, ...
                   {net_fn},{image_fn},coords);
  else
    res{1}{1} = tbar_cnn_infer_worker(...
        net_fn, image_fn, coords{1});
  end

  for ii=1:n_workers
    for jj=1:size(coords{ii},1)
      xs = coords{ii}(jj,1)+rf_offset(1)-1;
      xe = coords{ii}(jj,1)+coords{ii}(jj,4)-rf_offset(1);
      ys = coords{ii}(jj,2)+rf_offset(1)-1;
      ye = coords{ii}(jj,2)+coords{ii}(jj,5)-rf_offset(1);
      zs = coords{ii}(jj,3)+rf_offset(1)-1;
      ze = coords{ii}(jj,3)+coords{ii}(jj,6)-rf_offset(1);

      xi = xs:rf_stride(1):xe;
      yi = ys:rf_stride(1):ye;
      zi = zs:rf_stride(1):ze;
      for xo=0:rf_stride(1)-1
        for yo=0:rf_stride(1)-1
          for zo=0:rf_stride(1)-1
            out(xi+xo,yi+yo,zi+zo,:) = res{ii}{1}{jj};
          end
        end
      end
    end
  end

  if(exist('out_fn','var') && ~isempty(out_fn))
    if(exist(out_fn,'file'))
      delete(out_fn);
    end
    chunk_size = [30 30 30];
    chunk_size = min(chunk_size, im_sz);
    if(n_out > 1)
      chunk_size = [chunk_size 1];
    end
    h5create(out_fn,'/main',size(out),...
             'Datatype','single',...
             'Chunksize', chunk_size, ...
             'Deflate', 4, ...
             'Shuffle', 1);
    h5write(out_fn, '/main', out);
  end

  system(sprintf('rm %s', net_fn));
end
