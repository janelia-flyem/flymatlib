function tbar_cnn_full_infer_worker(net, work_dir, num_cubes)
% TBAR_CNN_FULL_INFER_WORKER full volume tbar inference on gpu
% TBAR_CNN_FULL_INFER_WORKER(net, work_dir, num_cubes)

  if(ischar(net)) % load net from file
    net = load(net);
    net = tbar_cnn_finalize_net(net.net);
  end
  num_gpus = 4; % start up gpus
  if(isempty(gcp('nocreate')))
    parpool('local',num_gpus);
    spmd, gpuDevice(labindex), end
  end

  while(true)
    % check for status/*.image
    dd = dir(sprintf('%s/status/*.image', work_dir));
    % if empty, pause, continue
    if(isempty(dd))
      % check for status/*.done, break if length==num_cubes
      dd = dir(sprintf('%s/status/*.done', work_dir));
      if(length(dd) == num_cubes)
        break
      else
        pause(10)
        continue
      end
    end

    % run infer on images/fn.h5
    [~,base_fn] = fileparts(dd(1).name);
    image_fn    = sprintf('%s/images/%s.h5', work_dir, base_fn);
    out_fn      = sprintf('%s/infer/%s.h5',  work_dir, base_fn);
    tbar_cnn_infer(net, image_fn, out_fn, 'gpu');

    % touch status/fn.infer, rm status/fn.image
    system(sprintf('touch %s/status/%s.infer', work_dir, base_fn));
    system(sprintf('rm %s/status/%s.image',    work_dir, base_fn));
  end
end
