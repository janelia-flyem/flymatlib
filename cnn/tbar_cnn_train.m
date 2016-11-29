function [net, info] = tbar_cnn_train(net, varargin)
% TBAR_CNN_TRAIN(net, varargin/opts)
%
% opts required fields:
%   expDir                  where to save models
%   imdbPath                training substack(s), validation
%   learningRate            also indicates number training epochs
%   nums                    samples per epoch
%   ratios                  class sampling ratios
%   classes                 class sampling classes
%
% opts optional fields:
%   weightDecay             defaults to 0.0005 (fml_cnn_train)
%   batchSize               defaults to 100
%   gpus                    defaults to empty (cpu)
%   is_autoencoder          defaults to false
%   data_aug (augmentation) defaults to false
%   aux_classes             auxillary constraints, default empty
%   patch_sz                only relevant for 2d autoencoder?


% required
opts.expDir              = [];
opts.imdbPath            = [];
opts.learningRate        = [];
opts.nums                = [];
opts.ratios              = [];
opts.classes             = [];

% optional
opts.weightDecay         = [];
opts.batchSize           = 100 ;
opts.gpus                = [];
opts.is_autoencoder      = false;
opts.data_aug            = false;
opts.aux_classes         = [];
opts.fraction_to_corrupt = [];
opts.zero_mask           = [];
opts.corrupt_type        = [];
opts.patch_sz            = [];
opts.label_sz            = [];
opts.use_all_data        = false;

[opts, varargin]         = vl_argparse(opts, varargin) ;
opts.train.learningRate  = opts.learningRate;
opts.train.weightDecay   = opts.weightDecay;
opts.train.batchSize     = opts.batchSize;
opts.train.gpus          = opts.gpus;

[opts, varargin]         = vl_argparse(opts, varargin) ;
opts.train.numEpochs     = numel(opts.train.learningRate) ;

opts.train.continue      = true ;
opts.train.expDir        = opts.expDir ;
opts.train.errorFunction = 'binary';
opts                     = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------

prefixes            = opts.imdbPath;
imdb.is_autoencoder = opts.is_autoencoder;
imdb.nums           = opts.nums;
imdb.ratios         = opts.ratios;
if(size(imdb.ratios,1) == 1)
  imdb.ratios = repmat(imdb.ratios, size(prefixes,1), 1);
end
imdb.classes        = opts.classes;
imdb.data_aug       = opts.data_aug;
imdb.use_all_data   = opts.use_all_data;
imdb.corrupt        = opts.fraction_to_corrupt;
imdb.zero_mask      = opts.zero_mask;
imdb.corrupt_type   = -1;
if(imdb.corrupt > 0)
  switch(opts.corrupt_type)
    case 'binomial'
      imdb.corrupt_type = 0;
    case 'gaussian'
      imdb.corrupt_type = 1;
    otherwise
      assert(false, 'FML:AssertionFailed', ...
             'unknown corruption type');
  end
end

if(imdb.is_autoencoder)
  opts.train.errorFunction = 'real';
end

net_info       = fml_simplenn_display(net);
imdb.patch_sz  = net_info.receptiveFieldSize(1,end);
imdb.ext_lower = floor( (imdb.patch_sz-1) / 2);
imdb.ext_upper = ceil(  (imdb.patch_sz-1) / 2);

if(imdb.is_autoencoder)
  imdb.patch_sz = opts.patch_sz;
  imdb.label_sz = opts.label_sz;
  imdb.is3d     = ~(imdb.is_autoencoder == 2);
end

if(size(prefixes,2)==3)
  has_aux     = true;
  aux_classes = opts.aux_classes;
else
  has_aux = false;
end

for ii=1:size(prefixes,1)
  data_fn   = prefixes{ii,1};
  labels_fn = sprintf('%slabels.h5', prefixes{ii,2});
  mask_fn   = sprintf('%smask.h5',   prefixes{ii,2});
  if(has_aux)
    aux_fn = prefixes{ii,3};
    if(~isempty(aux_fn))
      aa   = single(read_image_stack(aux_fn));
    else
      aa = [];
    end
  end

  imdb.data{ii} = read_image_stack(data_fn);

  if(~imdb.is_autoencoder)
    ll = int8(read_image_stack(labels_fn));
    if(size(ll,4)>1)
      opts.train.errorFunction = 'combobj';
    end
    mm = single(read_image_stack(mask_fn));

    mm([1:imdb.ext_lower,end-imdb.ext_upper+1:end],:,:) = 0;
    mm(:,[1:imdb.ext_lower,end-imdb.ext_upper+1:end],:) = 0;
    mm(:,:,[1:imdb.ext_lower,end-imdb.ext_upper+1:end]) = 0;
  end

  if(~imdb.is_autoencoder)
    for cc = 1:length(imdb.classes)
      % ll encodes neg/pos labels as 0/1
      cc_val = imdb.classes(cc);

      if(imdb.ratios(ii,cc) == 0)
        imdb.pts{ii,cc} = [];
        continue
      end

      if(~has_aux || isnan(aux_classes(cc)) || isempty(aa))
        if(isnan(cc_val)) % no constraint
          imdb.pts{ii,cc} = find( mm>0 );
        else % is valid neg/pos label
          imdb.pts{ii,cc} = find( (ll(:,:,:,1)==cc_val) & (mm>0) );
        end
      else
        aa_val = aux_classes(cc);
        if(isnan(cc_val)) % no constraint
          imdb.pts{ii,cc} = find(   mm>0  & (aa==aa_val) );
        else % is valid neg/pos label
          imdb.pts{ii,cc} = ...
              find( (ll(:,:,:,1)==cc_val) & (mm>0) & (aa==aa_val) );
        end
      end

    end
  end

  if(~imdb.is_autoencoder)
    clear mm;
    ll(:,:,:,1) = 2*ll(:,:,:,1)-1; % convert to -1,1
    imdb.labels{ii} = ll;
  end

  imdb.images.set = [1*ones(1,imdb.nums(1)), ...
                     2*ones(1,imdb.nums(2))];

end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

if(imdb.is_autoencoder)
  get_batch_func = @ae_get_batch;
else
  get_batch_func = @tbar_cnn_get_batch;
end

[net, info] = fml_cnn_train(net, imdb, get_batch_func, ...
    opts.train) ;
