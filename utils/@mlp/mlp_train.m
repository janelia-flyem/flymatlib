function mlp_train(this, num_features, num_points, ...
                   table_name, gm, gs, output_id, dfeval_dir, ...
                   max_job_id, mlp_core_obj) %#ok<INUSL>
  
  if(nargin == 1)
    mlp_unit_tests(this)
    return
  end

  nworkers_matlabpool = 6;
  if(mlp_core_obj.threaded)
    matlabpool('local', nworkers_matlabpool);
  end
  use_gpu_copy   = this.use_gpu;
  
  minibatch_size = this.minibatch_size_default;
  tmax           = minibatch_size*this.num_updates_default;
  num_updates    = floor(tmax/minibatch_size);
  num_sequences  = length(num_updates);
  
  pos_ratio_copy = this.pos_ratio;
  
  fprintf('use gpu: %d, pos_ratio: %s\n', ...
          use_gpu_copy, sprintf('%g, ', pos_ratio_copy));

  if(use_gpu_copy)
    [gpu_num, obtained_gpu] = get_gpu();
    fprintf('free mem: %g\n', obtained_gpu.FreeMemory);
    assert(gpu_num > 0, 'JANLIB:AssertionFailed', ...
           'could not get gpu');
  end

  % start from saved point if error on last run
  save_fn = mlp_save_name(table_name, dfeval_dir);
  if(exist(save_fn, 'file'))
    fprintf('restarting from saved state\n');
    
    saved_obj = load(save_fn);
    this.w1 = saved_obj.this.w1;
    this.b1 = saved_obj.this.b1;
    this.w2 = saved_obj.this.w2;
    this.b2 = saved_obj.this.b2;
    this.training_errsq    = saved_obj.this.training_errsq;
    this.training_classerr = saved_obj.this.training_classerr;
    
    eta_w   = saved_obj.eta_w;
    eta_b   = saved_obj.eta_b;
    t_start = saved_obj.t_start;
    u_start = saved_obj.u_start;
    
    if(use_gpu_copy)
      errsq    = cell(1, num_sequences);
      classerr = cell(1, num_sequences);
      for uu = 1:num_sequences
        errsq{uu}    = gpuArray(this.training_errsq{uu});
        classerr{uu} = gpuArray(this.training_classerr{uu});
      end
    else
      errsq    = this.training_errsq;
      classerr = this.training_classerr;
    end
  else
    eta_w = []; %this.eta_w_start;
    eta_b = []; %this.eta_b_start;
    t_start = 1;
    u_start = 1;
    
    errsq    = cell(1, num_sequences);
    classerr = cell(1, num_sequences);
    if(use_gpu_copy)
      for uu=1:num_sequences
        errsq{uu} = ...
            gpuArray.zeros([1,num_updates(uu)* ...
                            minibatch_size], 'single');
        classerr{uu} = ...
            gpuArray.zeros([1,num_updates(uu)* ...
                            minibatch_size], 'single');
      end
    else
      for uu=1:num_sequences
        errsq{uu}    = zeros(1,num_updates(uu)*minibatch_size);
        classerr{uu} = zeros(1,num_updates(uu)*minibatch_size);
      end
    end
  end
  eta_w_s = this.eta_w_start;
  eta_b_s = this.eta_b_start;
  if(~iscell(eta_w_s)), eta_w_s = {eta_w_s}; end
  if(~iscell(eta_b_s)), eta_b_s = {eta_b_s}; end
  
  assert(length(eta_w_s) == num_sequences, ...
         'JANCOM:AssertionFailed', ...
         sprintf('#learning rates %d ~= %d #update sequences', ...
                 length(eta_w_s), num_sequences));
  
  use_joint = 1;
  label_size = 1;
  if(exist('output_id','var') && ~isempty(output_id))
    if(isscalar(output_id))
      use_joint = 0;
    else
      label_size = length(output_id);
    end
  else
    output_id  = [];
  end

  num_instances = num_points;
  
  % adapt number to architecture
  mem_available = 3*10^9; % default if no GPU, cap at 3gb
  if(use_gpu_copy)
    mem_available = ...
        min(obtained_gpu.FreeMemory, 3*10^9);
  end
  mem_available = 0.25*mem_available; % additional correction
  num_to_pull = minibatch_size * floor( ...
    (mem_available / (4*num_features) - ...
     this.num_hidden(1) * minibatch_size) / ...
    minibatch_size);
  % additional correction for multiple jobs/node
  num_to_pull = floor(0.5 * num_to_pull);
  % cap to prevent from overloading db
  num_to_pull = min(num_to_pull, 1e5);
  
  num_subinstances = min(num_instances, num_to_pull);
  % num_subinstances = num_to_pull;
  fprintf('num_to_pull = %d\n', num_to_pull);
  
  assert(exist('table_name','var') > 0 && ...
         ~isempty(table_name), ...
         'JANLIB:AssertionFailed', ...
         'table name not provided');
  
  ec_tic = tic();
  [train_feats, train_labels, train_data, got_all] = ...
      mlp_core_obj.get_training_batch( ...
        num_subinstances, output_id, pos_ratio_copy);
  ec_toc = toc(ec_tic);
  fprintf('\tinitial pull: %d\n', ec_toc);
  if(got_all)
    fprintf('got all training data\n');
  end

  if(isempty(output_id))
    if(~iscell(train_labels))
      label_size = size(train_labels,2);
    else
      label_size = size(train_labels{1},2);
    end
  end
    
  % normalize
  if(use_gpu_copy)
    gm = gpuArray(single(gm));
    gs = gpuArray(single(gs));
    if(~iscell(train_feats))
      train_feats = gpuArray(train_feats);
    else
      for tt=1:length(train_feats)
        train_feats{tt} = gpuArray(train_feats{tt});
      end
    end
  end
  if(~iscell(train_feats))
    train_feats = bsxfun(@rdivide, ...
                         bsxfun(@minus, train_feats, gm), gs);
  else
    for tt=1:length(train_feats)
      train_feats{tt} = ...
          bsxfun(@rdivide, ...
                 bsxfun(@minus, train_feats{tt}, gm), gs);
    end
  end
  
  nhu                      = this.num_hidden;
  nhu_nlayers              = length(nhu);
  w1_mag_thresh_copy       = this.w1_mag_thresh;
  dropout_rate_input_copy  = this.dropout_rate_input;
  dropout_rate_hidden_copy = this.dropout_rate_hidden;
  if(isscalar(dropout_rate_hidden_copy))
    dropout_rate_hidden_copy = ...
        dropout_rate_hidden_copy * ones(size(nhu));
  end
  margin_copy              = this.margin;
  inv_margin_copy          = this.inv_margin;
  use_adagrad_copy         = this.use_adagrad;
  
  dampening_factor_copy    = this.dampening_factor;
  lambda_l2_copy           = this.lambda_l2;
  loss_func_copy           = this.loss_func;

  if(isscalar(lambda_l2_copy))
    lambda_l2_copy = lambda_l2_copy * ones(1,2);
  end
  if(length(lambda_l2_copy) ~= nhu_nlayers+1)
    lambda_l2_copy = ...
        [lambda_l2_copy(1)*ones(1,nhu_nlayers), ...
         lambda_l2_copy(2)];
  end
    
  % rng default % TODO: add option for turning this on?
  eps_init_copy = this.eps_init;
  if(isscalar(eps_init_copy))
    eps_init_copy = eps_init_copy*ones(2,1);
  end
  if(size(eps_init_copy,2) ~= nhu_nlayers+1)
    eps_init_copy = repmat(eps_init_copy, 1, nhu_nlayers+1);
  end
  w1c    = cell(1,nhu_nlayers);
  b1c    = cell(1,nhu_nlayers);
  if(isempty(this.w1) || isempty(this.w1{1}))
    w1c{1} = eps_init_copy(1,1)*randn(num_features, nhu(1));
    b1c{1} = eps_init_copy(2,1)*randn(nhu(1), 1);
  else
    w1c{1} = this.w1{1};
    b1c{1} = this.b1{1};
  end
  for nn = 2:nhu_nlayers
    if(isempty(this.w1) || isempty(this.w1{nn}))
      w1c{nn} = eps_init_copy(1,nn)*randn(nhu(nn-1), nhu(nn));
      b1c{nn} = eps_init_copy(2,nn)*randn(nhu(nn), 1);
    else
      w1c{nn} = this.w1{nn};
      b1c{nn} = this.b1{nn};
    end
  end
  if(isempty(this.w2))
    w2c = eps_init_copy(1,end)*randn(nhu(nhu_nlayers), label_size);
    b2c = eps_init_copy(2,end)*randn(label_size,1);
  else
    w2c = this.w2;
    b2c = this.b2;
  end
  
  w1d_hist = cell(1,nhu_nlayers);
  b1d_hist = cell(1,nhu_nlayers);
  for nn=1:nhu_nlayers
    w1d_hist{nn} = zeros(size(w1c{nn}));
    b1d_hist{nn} = zeros(size(b1c{nn}));
  end
  w2d_hist = zeros(size(w2c));
  b2d_hist = zeros(size(b2c));

  if(use_gpu_copy)
    w1_mag_thresh_copy = gpuArray(single(w1_mag_thresh_copy));

    for nn=1:nhu_nlayers
      w1c{nn} = gpuArray(single(w1c{nn}));
      b1c{nn} = gpuArray(single(b1c{nn}));
    end
    w2c = gpuArray(single(w2c));
    b2c = gpuArray(single(b2c));
  
    if(use_adagrad_copy)
      for nn=1:nhu_nlayers
        w1d_hist{nn} = gpuArray(single(w1d_hist{nn}));
        b1d_hist{nn} = gpuArray(single(b1d_hist{nn}));
      end
      w2d_hist = gpuArray(single(w2d_hist));
      b2d_hist = gpuArray(single(b2d_hist));
    end
    
    if(~iscell(train_labels))
      train_labels = gpuArray(single(train_labels));
    else
      for tt=1:length(train_feats)
        train_labels{tt} = gpuArray(train_labels{tt});
      end
    end
    %train_feats = gsingle(train_feats);
  end
  
  for nn=1:nhu_nlayers
    w1c{nn} = w1_normalize(w1c{nn}, w1_mag_thresh_copy);
  end
  
  sample_fractions    = -1*ones(1,label_size);
  sample_fractions(:) = pos_ratio_copy;
  sf_subset           = sample_fractions(sample_fractions > 0);
  n_sampled           = sum(sample_fractions > 0);
  if(n_sampled > 0)
    if(use_joint)
      minibatch_size_bal = ceil(minibatch_size/n_sampled) * ...
          reshape([(1-sf_subset); sf_subset], ...
                  1, 2*n_sampled);
    else
      minibatch_size_bal = minibatch_size * ...
          [(1-pos_ratio_copy), pos_ratio_copy];
    end
    minibatch_size_bal = round(minibatch_size_bal);
    minibatch_size = sum(minibatch_size_bal);
  end

  train_feats_str = ...
      sprintf('train_feats{%d}(i{%d},:) ; ', ...
              repmat(1:2*n_sampled, 2, 1));
  train_feats_str = ...
      sprintf('y0 = [%s]'';', train_feats_str);
  train_labels_str = ...
      sprintf('train_labels{%d}(i{%d},:) ; ', ...
              repmat(1:2*n_sampled, 2, 1));
  train_labels_str = ...
      sprintf('minibatch_labels = [%s]''; ', train_labels_str);
  
  ec_tic = tic();
  prev_sequence_partial = 0;
  for uu=u_start:num_sequences
    if(isempty(eta_w) || uu ~= u_start)
      eta_w = eta_w_s{uu};
      eta_b = eta_b_s{uu};
    end
    
    if(length(eta_w) ~= nhu_nlayers+1)
      eta_w = [eta_w(1)*ones(1,nhu_nlayers), eta_w(2)];
    end
    if(length(eta_b) ~= nhu_nlayers+1)
      eta_b = [eta_b(1)*ones(1,nhu_nlayers), eta_b(2)];
    end
    
    nhu_nlayers_sub = find(eta_w(1:end-1) < 0);
    if(isempty(nhu_nlayers_sub))
      nhu_nlayers_sub = nhu_nlayers;
    else
      nhu_nlayers_sub = min(nhu_nlayers_sub) - 1;
      assert(nhu_nlayers_sub > 0, ...
             'JANLIB:AssertionFailed', ...
             'invalid learning rate - nothing to learn');
    end
    if(nhu_nlayers_sub ~= nhu_nlayers || prev_sequence_partial > 0)
      w2c = eps_init_copy(1,end)*randn(nhu(nhu_nlayers_sub), ...
                                       label_size);
      b2c = eps_init_copy(2,end)*randn(label_size, 1);
      
      w2d_hist = zeros(size(w2c));
      b2d_hist = zeros(size(b2c));
    end

    fprintf('update %d/%d; sequence %d/%d; to layer %d/%d\n', ...
            t_start, num_updates(uu), uu, num_sequences, ...
            nhu_nlayers_sub, nhu_nlayers);

    for t=t_start:num_updates(uu)

      tm = mod(t-1, floor(num_subinstances/minibatch_size));
      if(n_sampled == 0)
        i = tm*minibatch_size+(1:minibatch_size);
        if(max(i)>num_subinstances)
          i = mod(i-1, num_subinstances)+1;
        end
        y0 = train_feats(i,:)';
      else
        i = cell(1,length(minibatch_size_bal));
        for jj=1:length(minibatch_size_bal)
          i{jj} = tm*minibatch_size_bal(jj) + ...
                  (1:minibatch_size_bal(jj));
          if(max(i{jj}) > size(train_feats{jj},1))
            i{jj} = mod(i{jj}-1, size(train_feats{jj},1)) + 1;
          end
        end
        if(use_joint)
          eval(train_feats_str);
        else
          y0 = [train_feats{1}(i{1},:) ;
                train_feats{2}(i{2},:)]';
        end
      end
      
      if(dropout_rate_input_copy > 0)
        if(use_gpu_copy)
          y0 = y0 .* (gpuArray.rand(size(y0)) > ...
                      dropout_rate_input_copy);
        else
          y0 = y0 .* (rand(size(y0)) > dropout_rate_input_copy);
        end
      end
      
      if(n_sampled == 0)
        minibatch_labels = train_labels(i,:)';
      else
        if(use_joint)
          eval(train_labels_str);
          % mb_size = size(minibatch_labels, 2);
          % fprintf('%g %g, %g %g, %g %g\n', ...
          %         sum(minibatch_labels(1,:)==0)/mb_size, ...
          %         sum(minibatch_labels(1,:)==1)/mb_size, ...
          %         sum(minibatch_labels(2,:)==0)/mb_size, ...
        else
          minibatch_labels = [train_labels{1}(i{1},:) ;
                              train_labels{2}(i{2},:)]';
        end
      end
      true_label = minibatch_labels > 0;
      label_mask = isnan(minibatch_labels);

      [w1d, b1d, w2d, b2d, errsq_iter, classerr_iter] = ...
          fwd_bwd_pass(w1c, b1c, w2c, b2c, ...
                       y0, true_label, label_mask, ...
                       minibatch_size, nhu_nlayers_sub, ...
                       use_joint, use_gpu_copy, ...
                       loss_func_copy, ...
                       dropout_rate_hidden_copy, ...
                       margin_copy, inv_margin_copy);

      tindex  = (t-1)*minibatch_size;
      tend    = t*minibatch_size;
      errsq{uu}(   tindex+1:tindex+minibatch_size) = ...
          errsq_iter;
      classerr{uu}(tindex+1:tindex+minibatch_size) = ...
          classerr_iter;

      
      if(rem(t,floor(5e4/minibatch_size))==0)
        if(0)%~isdeployed)
          plot(1:tend,cumsum(errsq{uu}(1:tend))./(1:tend));%#ok<UNRCH>
          ylim([0 0.2]);
          drawnow
        else
          errsq_vals = errsq{uu}(1:tend);
          errsq_vals(isinf(errsq_vals)) = [];
          fprintf('%g [%d/%d]\n', mean(errsq_vals), ...
                  t, num_updates(uu));
        end
      end
      
      if(rem(t,floor(5e5/minibatch_size))==0)
        % also save out current state
        copy_to_obj(this, w1c, b1c, w2c, b2c, ...
                    errsq, classerr, use_gpu_copy);
        save_mlp_state(this, eta_w, eta_b, t+1, uu, table_name, ...
                       dfeval_dir);
        fprintf('\t(saved)\n');
      end
      
      % dampen learning rate after every epoch
      if(rem(t, floor(num_instances/minibatch_size))==0)
        eta_w=max(0.0001, eta_w*dampening_factor_copy);
        eta_b=max(0.0001, eta_b*dampening_factor_copy);
        
        fprintf('new learning rate: %g [%d/%d]', ...
                eta_w(1), t, num_updates(uu));
        if(use_gpu_copy)
          fprintf(' (free mem: %g)', obtained_gpu.FreeMemory);
        end
        fprintf('\n');
        
        % display(['new learning rate: ', num2str(eta_w)]);      
      end
      
      if(use_adagrad_copy)
        for nn=1:nhu_nlayers_sub
          w1d_hist{nn} = w1d_hist{nn} + w1d{nn}.^2;
          b1d_hist{nn} = b1d_hist{nn} + b1d{nn}.^2;
          
          w1d{nn} = w1d{nn} ./ sqrt(w1d_hist{nn});
          b1d{nn} = b1d{nn} ./ sqrt(b1d_hist{nn});
        end
        
        w2d_hist = w2d_hist + w2d.^2;
        b2d_hist = b2d_hist + b2d.^2;
        
        w2d = w2d ./ sqrt(w2d_hist);
        b2d = b2d ./ sqrt(b2d_hist);
      end
      
      w2c = w2c + (eta_w(end) * w2d) - (lambda_l2_copy(end)*w2c);
      b2c = b2c + (eta_b(end) * b2d);
      for nn=1:nhu_nlayers_sub
        w1c{nn} = w1c{nn} + (eta_w(nn) * w1d{nn}) - ...
                  (lambda_l2_copy(nn)*w1c{nn});
        b1c{nn} = b1c{nn} + (eta_b(nn) * b1d{nn});
      end
      
      % constrain weight vector length
      %if rem(length(errsq), 100)==0
      for nn=1:nhu_nlayers_sub
        w1c{nn} = w1_normalize(w1c{nn}, w1_mag_thresh_copy);
      end
      %end

      if rem(tend,1e6)==0
        display(num2str(tend));
      end

      if(mod(t, floor(num_subinstances/minibatch_size)) == 0)

        if(~got_all)
          ec_toc = toc(ec_tic);
          fprintf('\tmlp time: %d\n', ec_toc);
          ec_tic = tic();
          [train_feats, train_labels] = ...
              mlp_core_obj.get_training_batch(...
                num_subinstances, output_id, pos_ratio_copy, ...
                train_data);
          ec_toc = toc(ec_tic);
          fprintf('\tpull time: %d\n', ec_toc);
          ec_tic = tic();
          
          if(use_gpu_copy)
            if(~iscell(train_feats))
              train_feats = gpuArray(train_feats);
              train_labels = gpuArray(train_labels);
            else
              for tt=1:length(train_feats)
                train_feats{tt}  = gpuArray(train_feats{tt});
                train_labels{tt} = gpuArray(train_labels{tt});
              end
            end
          end      
          % normalize
          if(~iscell(train_feats))
            train_feats = bsxfun(@rdivide, ...
                                 bsxfun(@minus, train_feats, gm), gs);
          else
            for tt=1:length(train_feats)
              train_feats{tt} = ...
                  bsxfun(@rdivide, ...
                         bsxfun(@minus, train_feats{tt}, gm), gs);
            end
          end
          
        else % got_all == true, permute order
          if(~iscell(train_feats))
            idx = randperm(size(train_feats,1));
            train_feats  = train_feats(idx,:);
            train_labels = train_labels(idx,:);
          else
            for tt=1:length(train_feats)
              idx = randperm(size(train_feats{tt},1));
              train_feats{tt}  = train_feats{tt}(idx,:);
              train_labels{tt} = train_labels{tt}(idx,:);
            end
          end
        end
        
      end
    end
    
    t_start = 1;
    if(nhu_nlayers_sub < nhu_nlayers)
      prev_sequence_partial = 1;
    else
      prev_sequence_partial = 0;
    end
  end
  
  copy_to_obj(this, w1c, b1c, w2c, b2c, ...
              errsq, classerr, use_gpu_copy);

  if(exist(save_fn, 'file'))
    system(sprintf('rm %s', save_fn));
  end
  if(use_gpu_copy)
    gpuDevice([]);
  end
  if(mlp_core_obj.threaded)
    matlabpool('close');
  end
end

function w = w1_normalize(w, v)
  sqsum = sum(w.*w,1);
  snorm = min(1,sqrt(v./sqsum));
  
  w = bsxfun(@times, w, snorm);
end

function copy_to_obj(this, w1c, b1c, w2c, b2c, ...
                     errsq, classerr, use_gpu_copy)
  if(use_gpu_copy)
    for i=1:length(w1c)
      this.w1{i} = gather(w1c{i});
      this.b1{i} = gather(b1c{i});
    end
    this.w2 = gather(w2c);
    this.b2 = gather(b2c);

    for uu=1:length(errsq)
      this.training_errsq{uu}    = gather(errsq{uu});
      this.training_classerr{uu} = gather(classerr{uu});
    end
  else
    this.w1 = w1c;
    this.b1 = b1c;
    this.w2 = w2c;
    this.b2 = b2c;
    
    this.training_errsq    = errsq;
    this.training_classerr = classerr;
  end
end

function save_mlp_state(this, eta_w, eta_b, t_start, ...
                        u_start, table_name, ...
                        dfeval_dir) %#ok<INUSL>
  save(mlp_save_name(table_name, dfeval_dir), ...
       'this', 'eta_w', 'eta_b', 't_start', 'u_start', '-v7.3');
end

function s = mlp_save_name(table_name, dfeval_dir) 
  
  s = sprintf('%s/%s_mlp_save.mat', ...
              dfeval_dir, table_name); 

  % doesn't work...
  % % actually, just save to scratch space
  % usr = getenv('USER');
  % s = sprintf('/scratch/%s/%s_mlp_save.mat', ...
  %             usr, table_name);

end

function [w1d, b1d, w2d, b2d, errsq, classerr] = ...
      fwd_bwd_pass(w1c, b1c, w2c, b2c, ...
                   y0, true_label, label_mask, ...
                   minibatch_size, nhu_nlayers_sub, ...
                   use_joint, use_gpu_copy, ...
                   loss_func_copy, ...
                   dropout_rate_hidden_copy, ...
                   margin_copy, inv_margin_copy)
  if(use_gpu_copy)
    [y1,y2] = thresh_linear_sigmoid_gpum(...
      y0, w1c(1:nhu_nlayers_sub), b1c(1:nhu_nlayers_sub), ...
      w2c, b2c, ...
      dropout_rate_hidden_copy(1:nhu_nlayers_sub));
  else
    [y1,y2] = thresh_linear_sigmoid(...
      y0, w1c(1:nhu_nlayers_sub), b1c(1:nhu_nlayers_sub), ...
      w2c, b2c, ...
      dropout_rate_hidden_copy(1:nhu_nlayers_sub));
  end
  
  pos =  max(0, 0.5 - y2 + margin_copy) ...
         - max(0, inv_margin_copy + y2 - 1);
  % pos = min(pos,  0.5);
  neg = -max(0, y2 - 0.5 + margin_copy) ...
        + max(0, inv_margin_copy - y2);
  % neg = max(neg, -0.5);
  pos(label_mask) = 0;
  neg(label_mask) = 0;
  label_mask_sum = sum(label_mask~=1,1);

  y2err   = ( true_label.*pos + ...
              ~true_label.*neg );
  if(loss_func_copy == 0)
    y2err = y2err .* y2 .* (1-y2);
  end
  
  if(use_joint)
    if(use_gpu_copy)
      w2d = sum(outer_products_gpum(...
        y1{end}, y2err), 3) / minibatch_size;
    else
      w2d = sum(outer_products(...
        y1{end}, y2err), 3) / minibatch_size;
    end
  else
    w2d   = y1{end} * y2err' / minibatch_size;
  end
  b2d     = mean(y2err,2);
  
  y1err      = cell(size(y1));
  y1err{end} = (y1{end}>0) .* (w2c * y2err);
  w1d        = cell(size(y1));
  b1d        = cell(size(y1));
  
  for nn=nhu_nlayers_sub:-1:2
    b1d{nn} = mean(y1err{nn},2);
    if(use_gpu_copy)
      w1d{nn} = sum(outer_products_gpum(...
        y1{nn-1}, y1err{nn}), 3) / minibatch_size;
    else
      w1d{nn} = sum(outer_products(...
        y1{nn-1}, y1err{nn}), 3) / minibatch_size;
    end
    y1err{nn-1} = (y1{nn-1}>0) .* (w1c{nn} * y1err{nn});
  end
  
  b1d{1}   = mean(y1err{1},2);
  if(use_gpu_copy)
    w1d{1} = sum(outer_products_gpum(...
      y0, y1err{1}), 3) / minibatch_size;
  else
    w1d{1} = sum(outer_products(...
      y0, y1err{1}), 3) / minibatch_size;
  end
  
  if(loss_func_copy == 0)
    errsq    = 0.5*(sum( ( true_label.*pos.*pos ) + ...
                         (~true_label.*neg.*neg ), 1))';
  else
    ce_val   = true_label.*log(y2) + ~true_label.*log(1-y2);
    ce_val(isnan(ce_val)) = 0;
    errsq    = -sum( ce_val , 1)';
  end
  classerr = ( sum((y2>0.5)==true_label,1) ./ ...
               label_mask_sum )';

end

function mlp_unit_tests(this)
  rng default
  fprintf('running mlp unit tests\n');
  
  num_features = 100;
  label_size   = 3;
  
  use_gpu_copy = this.use_gpu;
  if(use_gpu_copy)
    [gpu_num, obtained_gpu] = get_gpu();
    fprintf('free mem: %g\n', obtained_gpu.FreeMemory);
    assert(gpu_num > 0, 'JANLIB:AssertionFailed', ...
           'could not get gpu');
  end

  minibatch_size = this.minibatch_size_default;
  
  nhu         = this.num_hidden;
  nhu_nlayers = length(nhu);
  
  eps_init_copy = this.eps_init;
  if(isscalar(eps_init_copy))
    eps_init_copy = eps_init_copy*ones(2,1);
  end
  if(size(eps_init_copy,2) ~= nhu_nlayers+1)
    eps_init_copy = repmat(eps_init_copy, 1, nhu_nlayers+1);
  end
  
  w1c    = cell(1,nhu_nlayers);
  b1c    = cell(1,nhu_nlayers);
  w1c{1} = eps_init_copy(1,1)*randn(num_features, nhu(1));
  b1c{1} = eps_init_copy(2,1)*randn(nhu(1), 1);
  for nn = 2:nhu_nlayers
    w1c{nn} = eps_init_copy(1,nn)*randn(nhu(nn-1), nhu(nn));
    b1c{nn} = eps_init_copy(2,nn)*randn(nhu(nn), 1);
  end
  w2c = eps_init_copy(1,end)*randn(nhu(nhu_nlayers), label_size);
  b2c = eps_init_copy(2,end)*randn(label_size,1);
  
  if(use_gpu_copy)
    for nn=1:nhu_nlayers
      w1c{nn} = gpuArray(double(w1c{nn}));
      b1c{nn} = gpuArray(double(b1c{nn}));
    end
    w2c = gpuArray(double(w2c));
    b2c = gpuArray(double(b2c));
  end
  
  y0         = randn(num_features, minibatch_size);
  true_label = rand( label_size,   minibatch_size) > 0.5;
  label_mask = false(size(true_label));
  
  if(use_gpu_copy)
    y0         = gpuArray(double(y0));
    true_label = gpuArray(double(true_label));
    label_mask = gpuArray(label_mask);
  end

  [w1d, b1d, w2d, b2d] = fwd_bwd_pass(...
    w1c, b1c, w2c, b2c, ...
    y0, true_label, label_mask, ...
    minibatch_size, nhu_nlayers, 1, use_gpu_copy, ...
    this.loss_func, zeros(1, nhu_nlayers+1), ...
    this.margin, this.inv_margin);
  
  mlp_eps = 1e-5;

  n_test = 20;
  
  for ii=1:n_test
    w2c(ii) = w2c(ii) + mlp_eps;
    [~,~,~,~, errsq1] = fwd_bwd_pass(...
      w1c, b1c, w2c, b2c, ...
      y0, true_label, label_mask, ...
      minibatch_size, nhu_nlayers, 1, use_gpu_copy, ...
      this.loss_func, zeros(1, nhu_nlayers+1), ...
      this.margin, this.inv_margin);
    w2c(ii) = w2c(ii) - 2*mlp_eps;
    [~,~,~,~, errsq2] = fwd_bwd_pass(...
      w1c, b1c, w2c, b2c, ...
      y0, true_label, label_mask, ...
      minibatch_size, nhu_nlayers, 1, use_gpu_copy, ...
      this.loss_func, zeros(1, nhu_nlayers+1), ...
      this.margin, this.inv_margin);
    w2c(ii) = w2c(ii) + mlp_eps;
    
    ff = [(sum(errsq1)-sum(errsq2))/(2*mlp_eps*minibatch_size), ...
          -w2d(ii)];
    fprintf('%g\t%g\t%g\n', ff(1), ff(2), ff(1)/ff(2));
  end
  fprintf('\n');
  
  for ii=1:label_size
    b2c(ii) = b2c(ii) + mlp_eps;
    [~,~,~,~, errsq1] = fwd_bwd_pass(...
      w1c, b1c, w2c, b2c, ...
      y0, true_label, label_mask, ...
      minibatch_size, nhu_nlayers, 1, use_gpu_copy, ...
      this.loss_func, zeros(1, nhu_nlayers+1), ...
      this.margin, this.inv_margin);
    b2c(ii) = b2c(ii) - 2*mlp_eps;
    [~,~,~,~, errsq2] = fwd_bwd_pass(...
      w1c, b1c, w2c, b2c, ...
      y0, true_label, label_mask, ...
      minibatch_size, nhu_nlayers, 1, use_gpu_copy, ...
      this.loss_func, zeros(1, nhu_nlayers+1), ...
      this.margin, this.inv_margin);
    b2c(ii) = b2c(ii) + mlp_eps;
    
    ff = [(sum(errsq1)-sum(errsq2))/(2*mlp_eps*minibatch_size), ...
          -b2d(ii)];
    fprintf('%g\t%g\t%g\n', ff(1), ff(2), ff(1)/ff(2));
  end
  fprintf('\n');

  for jj=1:nhu_nlayers
    for ii=1:n_test
      w1c{jj}(ii) = w1c{jj}(ii) + mlp_eps;
      [~,~,~,~, errsq1] = fwd_bwd_pass(...
        w1c, b1c, w2c, b2c, ...
        y0, true_label, label_mask, ...
        minibatch_size, nhu_nlayers, 1, use_gpu_copy, ...
        this.loss_func, zeros(1, nhu_nlayers+1), ...
        this.margin, this.inv_margin);
      w1c{jj}(ii) = w1c{jj}(ii) - 2*mlp_eps;
      [~,~,~,~, errsq2] = fwd_bwd_pass(...
        w1c, b1c, w2c, b2c, ...
        y0, true_label, label_mask, ...
        minibatch_size, nhu_nlayers, 1, use_gpu_copy, ...
        this.loss_func, zeros(1, nhu_nlayers+1), ...
        this.margin, this.inv_margin);
      w1c{jj}(ii) = w1c{jj}(ii) + mlp_eps;
      
      ff = [(sum(errsq1)-sum(errsq2))/(2*mlp_eps*minibatch_size), ...
            -w1d{jj}(ii)];
      fprintf('%g\t%g\t%g\n', ff(1), ff(2), ff(1)/ff(2));
    end
    fprintf('\n');
    
    for ii=1:n_test
      b1c{jj}(ii) = b1c{jj}(ii) + mlp_eps;
      [~,~,~,~, errsq1] = fwd_bwd_pass(...
        w1c, b1c, w2c, b2c, ...
        y0, true_label, label_mask, ...
        minibatch_size, nhu_nlayers, 1, use_gpu_copy, ...
        this.loss_func, zeros(1, nhu_nlayers+1), ...
        this.margin, this.inv_margin);
      b1c{jj}(ii) = b1c{jj}(ii) - 2*mlp_eps;
      [~,~,~,~, errsq2] = fwd_bwd_pass(...
        w1c, b1c, w2c, b2c, ...
        y0, true_label, label_mask, ...
        minibatch_size, nhu_nlayers, 1, use_gpu_copy, ...
        this.loss_func, zeros(1, nhu_nlayers+1), ...
        this.margin, this.inv_margin);
      b1c{jj}(ii) = b1c{jj}(ii) + mlp_eps;
      
      ff = [(sum(errsq1)-sum(errsq2))/(2*mlp_eps*minibatch_size), ...
            -b1d{jj}(ii)];
      fprintf('%g\t%g\t%g\n', ff(1), ff(2), ff(1)/ff(2));
    end
    fprintf('\n');
  end
  
  if(use_gpu_copy)
    gpuDevice([]);
  end
  
end
