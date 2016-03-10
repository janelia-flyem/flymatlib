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
opts.corrupt_type        = [];
opts.patch_sz            = [];

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
imdb.classes        = opts.classes;
imdb.data_aug       = opts.data_aug;
imdb.corrupt        = opts.fraction_to_corrupt;
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

if(imdb.is_autoencoder == 2)
  imdb.patch_sz = opts.patch_sz;
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
    aa     = single(read_image_stack(aux_fn));
  end
  
  imdb.data{ii} = read_image_stack(data_fn);
  
  ll = single(read_image_stack(labels_fn));
  mm = single(read_image_stack(mask_fn));
  
  mm([1:imdb.ext_lower,end-imdb.ext_upper+1:end],:,:) = 0;
  mm(:,[1:imdb.ext_lower,end-imdb.ext_upper+1:end],:) = 0;
  mm(:,:,[1:imdb.ext_lower,end-imdb.ext_upper+1:end]) = 0;

  if(imdb.is_autoencoder ~= 2)
    for cc = 1:length(imdb.classes)
      % ll encodes neg/pos labels as 0/1
      cc_val = imdb.classes(cc);

      if(~has_aux || isnan(aux_classes(cc)) )
        if(isnan(cc_val)) % no constraint
          imdb.pts{ii,cc} = find( mm>0 );
        else % is valid neg/pos label
          imdb.pts{ii,cc} = find( (ll==cc_val) & (mm>0) );
        end
      else
        aa_val = aux_classes(cc);
        if(isnan(cc_val)) % no constraint
          imdb.pts{ii,cc} = find(   mm>0  & (aa==aa_val) );
        else % is valid neg/pos label
          imdb.pts{ii,cc} = ...
              find( (ll==cc_val) & (mm>0) & (aa==aa_val) );
        end
      end
      
    end
  end
    
  clear ll mm
  
  imdb.images.set = [1*ones(1,imdb.nums(1)), ...
                     2*ones(1,imdb.nums(2))];

end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

if(imdb.is_autoencoder == 2)
  get_batch_func = @getBatch_autoencoder2d;
else
  get_batch_func = @getBatch_on_the_fly;
end

[net, info] = fml_cnn_train(net, imdb, get_batch_func, ...
    opts.train) ;

% --------------------------------------------------------------------
function [data, labels] = getBatch_autoencoder2d(imdb, batch)
% --------------------------------------------------------------------
ii = 1; % training
if(min(batch) > imdb.nums(1))
  ii = 2; % validation
end

n_examples = length(batch);

patch_sz     = imdb.patch_sz;
corrupt_type = imdb.corrupt_type;

if(corrupt_type == 0)
  num_corrupt = floor(patch_sz^2 * imdb.corrupt);
end
if(corrupt_type == 1)
  sgm_corrupt = imdb.corrupt;
end

d_sz = size(imdb.data{ii});
xx = ceil( (d_sz(1)-patch_sz+1)*rand(n_examples,1) );
yy = ceil( (d_sz(2)-patch_sz+1)*rand(n_examples,1) );
zz = ceil( (d_sz(3))           *rand(n_examples,1) );

data   = zeros(patch_sz,patch_sz,1,...
               n_examples, 'single');
labels = zeros(patch_sz,patch_sz,1,...
               n_examples, 'single');

if(imdb.data_aug)
  aug_rot = floor(4*rand(n_examples,1));
  aug_ref = floor(2*rand(n_examples,1));
else
  aug_rot = zeros(n_examples,1);
  aug_ref = zeros(n_examples,1);
end

for jj=1:n_examples
  data_tmp = imdb.data{ii}(...
    xx(jj):xx(jj)+patch_sz-1,...
    yy(jj):yy(jj)+patch_sz-1,...
    zz(jj));
  if(aug_rot(jj))
    for kk=1:size(data_tmp,3)
      data_tmp(:,:,kk) = rot90( data_tmp(:,:,kk),aug_rot(jj));
    end
  end
  if(aug_ref(jj))
    for kk=1:size(data_tmp,3)
      data_tmp(:,:,kk) = fliplr(data_tmp(:,:,kk));
    end
  end
  
  labels(:,:,1,jj) = data_tmp;
  if(corrupt_type == 0)
    rr = randsample(patch_sz^2, num_corrupt);
    data_tmp(rr) = 0;
  end
  if(corrupt_type == 1)
    data_tmp = data_tmp + ...
        sgm_corrupt * randn(patch_sz,patch_sz);
  end
  data(:,:,1,jj) = data_tmp;
end


% --------------------------------------------------------------------
function [data, labels] = getBatch_on_the_fly(imdb, batch)
% --------------------------------------------------------------------
n_substacks = length(imdb.data);
iiss = 1:(n_substacks-1); % training
if(min(batch) > imdb.nums(1))
  iiss = n_substacks; % validation
end

patch_sz  = imdb.patch_sz;
ext_lower = imdb.ext_lower;
ext_upper = imdb.ext_upper;

corrupt_type = imdb.corrupt_type;

if(corrupt_type == 0)
  num_corrupt = floor(patch_sz^3 * imdb.corrupt);
end
if(corrupt_type == 1)
  sgm_corrupt = imdb.corrupt;
end

n_examples_total = length(batch);
data   = zeros(patch_sz,patch_sz,patch_sz,1,...
               n_examples_total, 'single');
labels = ones(1,1,1, 1,...
              n_examples_total, 'single');
idx = 0;

n_examples = n_examples_total/length(iiss);
assert(n_examples == round(n_examples), ...
       'FLYEMLIB:AssertionFailed',...
       ['number of training substacks should ' ...
        'evenly divide batch size']);

num_tot  = 0;
n_ratios      = length(imdb.classes);
num_per_class = zeros(1,n_ratios);
for cc = 1:n_ratios-1
  num_per_class(cc) = floor(n_examples * imdb.ratios(cc));
  num_tot = num_tot + num_per_class(cc);
end
num_per_class(n_ratios) = n_examples - num_tot;

for ii=iiss
  for cc = 1:n_ratios
    pts = randsample(imdb.pts{ii, cc}, num_per_class(cc), true);
    %length(pts)<num_per_class(cc+1));
  
    [xa,ya,za] = ind2sub(size(imdb.data{ii}), pts);

    n_pts = length(pts);
    for jj = 1:n_pts
      idx = idx + 1;
      xx = xa(jj); yy = ya(jj); zz = za(jj);
      
      data_tmp = imdb.data{ii}(...
          xx-ext_lower:xx+ext_upper,...
          yy-ext_lower:yy+ext_upper,...
          zz-ext_lower:zz+ext_upper);
      if(corrupt_type == 0)
        rr = randsample(patch_sz^3, num_corrupt);
        data_tmp(rr) = 0;
      end
      if(corrupt_type == 1)
        data_tmp = data_tmp + ...
            sgm_corrupt * randn(patch_sz,patch_sz,patch_sz);
      end
      data(:,:,:,1,idx) = data_tmp;
      if(imdb.is_autoencoder)
        labels(1,1,1,1,idx) = imdb.data{ii}(xx,yy,zz);
      else
        % warning: this will be incorrect if imdb.classes is not valid
        labels(1,1,1,1,idx) = 2*imdb.classes(cc)-1;
      end
    end
  end
end

if(imdb.data_aug)
  nn = size(data,5);
  aug_rot = floor(4*rand(nn,1));
  aug_ref = floor(2*rand(nn,1));

  for ii=1:nn
    if(aug_rot(ii))
      for zz=1:size(data,3)
        data(:,:,zz,1,ii) = rot90( data( :,:,zz,1,ii),aug_rot(ii));
      end
    end
    if(aug_ref(ii))
      for zz=1:size(data,3)
        data(:,:,zz,1,ii) = fliplr(data(:,:,zz,1,ii));
      end
    end
  end
end
