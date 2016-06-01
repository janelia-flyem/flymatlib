function out = tbar_cnn_infer_worker(net_fn, image_fn, coords, ...
                                     use_gpu)
% out = TBAR_CNN_INFER_WORKER(net_fn, image_fn, coords)
% performs inference with given net, image
%   net_fn can be net in saved .mat or net itself

  if(ischar(net_fn))
    net = load(net_fn);
    net = net.net;
  else
    net = net_fn;
  end
  if(~exist('use_gpu','var') || isempty(use_gpu))
    use_gpu = false;
  end

  if(use_gpu)
    net = fml_simplenn_move(net, 'gpu');
  end

  n_to_process = size(coords,1);
  out = cell(1,n_to_process);

  if(use_gpu)
    for ii=1:n_to_process
      im  = gpuArray(read_image_stack(...
          image_fn, [], coords(ii,1:3), coords(ii,4:6)));

      res = fml_simplenn(net, im, [],[], 'mode', 'test');
      out{ii} = gather(res(end).x);
    end
  else
    for ii=1:n_to_process
      im  = read_image_stack(...
          image_fn, [], coords(ii,1:3), coords(ii,4:6));

      res = fml_simplenn(net, im, [],[], 'mode', 'test');
      out{ii} = res(end).x;
    end
  end
end
