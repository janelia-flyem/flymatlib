function [net, info] = tbar_cnn_train_online(...
    net, last_epoch, num_ss_per_epoch, prev_wght, opts)

  if(~exist('prev_wght','var') || isempty(prev_wght))
    prev_wght = 0;
  end

  num_epochs = length(opts.learningRate);
  n_train_ss = size(opts.imdbPath,1) - 1;
  train_idx  = 0;
  prefixes   = opts.imdbPath;

  net_info       = fml_simplenn_display(net);
  imdb.patch_sz  = net_info.receptiveFieldSize(1,end);
  imdb.ext_lower = floor( (imdb.patch_sz-1) / 2);
  imdb.ext_upper = ceil(  (imdb.patch_sz-1) / 2);

  opts_sub               = opts;
  opts_sub.nums          = num_ss_per_epoch * opts.nums;
  opts_sub.ratios        = opts.ratios([...
      ones(1,num_ss_per_epoch), 2],:);

  for ii = (last_epoch+1):num_epochs
    tt_s = tic;
    train_idx = train_idx(end) + (1:num_ss_per_epoch);
    train_idx = mod(train_idx - 1, n_train_ss) + 1;

    net_infer = load(sprintf(...
        '%s/net-epoch-%d.mat', opts.expDir, ii-1));
    net_infer = tbar_cnn_finalize_net(net_infer.net);

    n_samples = 0;
    for jj=1:length(train_idx)
      % preliminaries
      tr_idx_ss = train_idx(jj);
      data_fn   = prefixes{tr_idx_ss,1};
      labels_fn = sprintf('%slabels.h5', prefixes{tr_idx_ss,2});
      mask_fn   = sprintf('%smask.h5',   prefixes{tr_idx_ss,2});
      ll{jj}    = read_image_stack(labels_fn);
      mm        = read_image_stack(mask_fn);
      mm([1:imdb.ext_lower,end-imdb.ext_upper+1:end],:,:) = 0;
      mm(:,[1:imdb.ext_lower,end-imdb.ext_upper+1:end],:) = 0;
      mm(:,:,[1:imdb.ext_lower,end-imdb.ext_upper+1:end]) = 0;

      ll{jj} = (ll{jj} & mm);
      n_samples = n_samples + nnz(ll{jj});

      % run inference
      out    = tbar_cnn_infer(net_infer, data_fn, [], 'gpu');
      % compute loss for negatives only
      cc{jj} = (mm==1).*(-(ll{jj}==0).*log(max(1-out,1e-5)));

      neg_wght(jj) = sum(cc{jj}(:));
    end

    % n_samples_neg = round(n_samples * neg_wght / sum(neg_wght));
    cc_tol = 0.1;
    for jj=1:length(train_idx)
      cc_sub{jj,1} = cc{jj}(cc{jj}>cc_tol);
      cc_sz(jj)    = size(cc_sub{jj},1);
    end
    cc_sub_all = cell2mat(cc_sub);
    cc_sz_cum  = cumsum(cc_sz);
    [~,cc_idx] = sort(cc_sub_all);
    cc_idx = cc_idx(max(1,end-n_samples+1):end);
    for jj=1:length(train_idx)
      n_samples_neg(jj) = sum(cc_idx <= cc_sz_cum(jj));
    end
    n_samples_neg(2:end) = diff(n_samples_neg);
    n_samples_neg        = max( n_samples_neg, 1000);

    if(~exist('n_samples_neg_old','var'))
      n_samples_neg_old = round(sum(...
          n_samples_neg) / num_ss_per_epoch) * ones(...
              1,size(prefixes,1));
    end
    n_samples_neg                = round(...
        (1-prev_wght) * n_samples_neg + ...
           prev_wght  * n_samples_neg_old(train_idx));
    n_samples_neg_old(train_idx) = n_samples_neg;

    for jj=1:length(train_idx)
      cc_idx1 = (ll{jj}==1);
      cc_min  = 0.001;
      cc_idx0 = find(cc{jj}>cc_min);
      cc_idx0 = randsample(cc_idx0, n_samples_neg(jj), true, ...
                           cc{jj}(cc_idx0));

      idx1      = nnz(cc_idx1);
      idx0      = length(cc_idx0);
      ww_ww(jj) = idx0+idx1;
      fprintf('\t%d\t%d\n', idx0, idx1);
      ww_counts(ii,jj,1) = idx0;
      ww_counts(ii,jj,2) = idx1;

      % write out to wght_fn
      wght_fn{jj} = sprintf(...
          '%s/wght%02d.h5', opts.expDir, train_idx(jj));
      cc_sz = size(cc{jj});
      if(~exist(wght_fn{jj},'file'))
        chunk_sz = min([50 50 50], cc_sz);
        h5create(wght_fn{jj}, '/main', cc_sz, ...
                 'Datatype', 'uint8', ...
                 'ChunkSize', chunk_sz, ...
                 'Deflate', 4, ...
                 'Shuffle', 1);
      end
      ww = zeros(cc_sz, 'uint8');
      ww(cc_idx1) = 1;
      ww(cc_idx0) = 1;
      h5write(wght_fn{jj}, '/main', ww);
    end

    % set wght_fn
    opts_sub.imdbPath      = opts.imdbPath([train_idx,end],:);
    for jj=1:num_ss_per_epoch
      opts_sub.imdbPath{jj,3} = wght_fn{jj};
    end
    opts_sub.learningRate  = opts.learningRate(1:ii);
    ww_ww = ww_ww / sum(ww_ww);
    opts_sub.ratios(1:num_ss_per_epoch,1) = ww_ww;
    tm1 = toc(tt_s);
    save(sprintf('%s/ww_counts.mat', opts.expDir), ...
         'ww_counts');
    % keyboard

    % run
    tt_s = tic;
    [net, info] = tbar_cnn_train(net, opts_sub);
    tm2 = toc(tt_s);

    fprintf('%g, %g\n', tm1, tm2);
  end
  info.ww_counts = ww_counts;
end
