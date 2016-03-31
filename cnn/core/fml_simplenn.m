function res = fml_simplenn(net, x, dzdy, res, varargin)
% fork of VL_SIMPLENN with appropriate handling of
%   3d convolutions, max-pooling, batch normalization w/3d data
%
%VL_SIMPLENN  Evaluate a SimpleNN network.
%   RES = VL_SIMPLENN(NET, X) evaluates the convnet NET on data X.
%   RES = VL_SIMPLENN(NET, X, DZDY) evaluates the convnent NET and its
%   derivative on data X and output derivative DZDY (foward+bacwkard pass).
%   RES = VL_SIMPLENN(NET, X, [], RES) evaluates the NET on X reusing the
%   structure RES.
%   RES = VL_SIMPLENN(NET, X, DZDY, RES) evaluates the NET on X and its
%   derivatives reusing the structure RES.
%
%   This function process networks using the SimpleNN wrapper
%   format. Such networks are 'simple' in the sense that they consist
%   of a linear sequence of computational layers. You can use the
%   `dagnn.DagNN` wrapper for more complex topologies, or write your
%   own wrapper around MatConvNet computational blocks for even
%   greater flexibility.
%
%   The format of the network structure NET and of the result
%   structure RES are described in some detail below. Most networks
%   expect the input data X to be standardized, for example by
%   rescaling the input image(s) and subtracting a mean. Doing so is
%   left to the user, but information on how to do this is usually
%   contained in the `net.meta` field of the NET structure (see
%   below).
%
%   The NET structure needs to be updated as new features are
%   introduced in MatConvNet; use the `VL_SIMPLENN_TIDY()` function
%   to make an old network current, as well as to cleanup and check
%   the structure of an existing network.
%
%   Networks can run either on the CPU or GPU. Use VL_SIMPLENN_MOVE()
%   to move the network parameters between these devices.
%
%   To print or obtain summary of the network structure, use the
%   VL_SIMPLENN_DISPLAY() function.
%
%   VL_SIMPLENN(NET, X, DZDY, RES, 'OPT', VAL, ...) takes the following
%   options:
%
%   `Mode`:: `normal`
%      Specifies the mode of operation. It can be either `'normal'` or
%      `'test'`. In test mode, dropout and batch-normalization are
%      bypassed. Note that, when a network is deployed, it may be
%      preferable to *remove* such blocks altogether.
%
%   `ConserveMemory`:: `true`
%      Aggressively delete intermediate results. This in practice has
%      a very small performance hit and allows training much larger
%      models. However, it can be useful to disable it for
%      debugging. It is also possible to preserve individual layer outputs
%      by setting `net.layers{...}.precious` to `true`.
%
%   `CuDNN`:: `true`
%      Use CuDNN when available.
%
%   `Accumulate`:: `false`
%      Accumulate gradients in back-propagation instead of rewriting
%      them. This is useful to break the computation in sub-batches.
%      The gradients are accumulated to the provided RES structure
%      (i.e. to call VL_SIMPLENN(NET, X, DZDY, RES, ...).
%
%   `SkipForward`:: `false`
%      Reuse the output values from the provided RES structure and compute
%      only the derivatives (bacward pass).
%
%   ## The result format
%
%   SimpleNN returns the result of its calculations in the RES
%   structure array. RES(1) contains the input to the network, while
%   RES(2), RES(3), ... contain the output of each layer, from first
%   to last. Each entry has the following fields:
%
%   - `res(i+1).x`: the output of layer `i`. Hence `res(1).x` is the
%     network input.
%
%   - `res(i+1).aux`: any auxiliary output data of layer i. For example,
%     dropout uses this field to store the dropout mask.
%
%   - `res(i+1).dzdx`: the derivative of the network output relative
%     to the output of layer `i`. In particular `res(1).dzdx` is the
%     derivative of the network output with respect to the network
%     input.
%
%   - `res(i+1).dzdw`: a cell array containing the derivatives of the
%     network output relative to the parameters of layer `i`. It can
%     be a cell array for multiple parameters.
%
%   ## The network format
%
%   The network is represented by the NET structure, which contains
%   two fields:
%
%   - `net.layers` is a cell array with the CNN layers.
%
%   - `net.meta` is a grab-bag of auxiliary application-dependent
%     information, including for example details on how to normalize
%     input data, the class names for a classifiers, or details of
%     the learning algorithm. The content of this field is ignored by
%     VL_SIMPLENN().
%
%   SimpleNN is aware of the following layers:
%
%   Convolution layer::
%     The convolution layer wraps VL_NNCONV(). It has fields:
%
%     - `layer.type` contains the string `'conv'`.
%     - `layer.weights` is a cell array with filters and biases.
%     - `layer.stride` is the sampling stride (e.g. 1).
%     - `layer.pad` is the padding (e.g. 0).
%
%   Convolution transpose layer::
%     The convolution transpose layer wraps VL_NNCONVT(). It has fields:
%
%     - `layer.type` contains the string `'convt'`.
%     - `layer.weights` is a cell array with filters and biases.
%     - `layer.upsample` is the upsampling factor (e.g. 1).
%     - `layer.crop` is the amount of output cropping (e.g. 0).
%
%   Max pooling layer::
%     The max pooling layer wraps VL_NNPOOL(). It has fields:
%
%     - `layer.type` contains the string `'pool'`.
%     - `layer.method` is the pooling method (either 'max' or 'avg').
%     - `layer.pool` is the pooling size (e.g. 3).
%     - `layer.stride` is the sampling stride (usually 1).
%     - `layer.pad` is the padding (usually 0).
%
%   Normalization (LRN) layer::
%     The normalization layer wraps VL_NNNORMALIZE(). It has fields:
%
%     - `layer.type` contains the string `'normalize'` or `'lrn'`.
%     - `layer.param` contains the normalization parameters (see VL_NNNORMALIZE()).
%
%   Spatial normalization layer::
%     The spatial normalization layer wraps VL_NNSPNORM(). It has fields:
%
%     - `layer.type` contains the string `'spnorm'`.
%     - `layer.param` contains the normalization parameters (see VL_NNSPNORM()).
%
%   Batch normalization layer::
%     This layer wraps VL_NNBNORM(). It has fields:
%
%     - `layer.type` contains the string `'bnorm'`.
%     - `layer.weights` contains is a cell-array with, multiplier and
%       biases, and moments parameters
%
%     Note that moments are used only in `'test'` mode to bypass batch
%     normalization.
%
%   ReLU and Sigmoid layers::
%     The ReLU layer wraps VL_NNRELU(). It has fields:
%
%     - `layer.type` contains the string `'relu'`.
%     - `layer.leak` is the leak factor (e.g. 0).
%
%     The sigmoid layer is the same, but for the sigmoid function,
%     with `relu` replaced by `sigmoid` and no leak factor.
%
%   Dropout layer::
%     The dropout layer wraps VL_NNDROPOUT(). It has fields:
%
%     - `layer.type` contains the string `'dropout'`.
%     - `layer.rate` is the dropout rate (e.g. 0.5).
%
%     Note that the block is bypassed in `test` mode.
%
%   Softmax layer::
%     The softmax layer wraps VL_NNSOFTMAX(). It has fields
%
%     - `layer.type` contains the string`'softmax'`.
%
%   Log-loss layer and softmax-log-loss::
%     The log-loss layer wraps VL_NNLOSS(). It has fields:
%
%     - `layer.type` contains `'loss'`.
%     - `layer.class` contains the ground-truth class labels.
%
%     The softmax-log-loss layer wraps VL_NNSOFTMAXLOSS() instead. it
%     has the same parameters, but `type` contains the `'softmaxloss'`
%     string.
%
%   P-dist layer::
%     The p-dist layer wraps VL_NNPDIST(). It has fields:
%
%     - `layer.type` contains the string  `'pdist'`.
%     - `layer.p` is the P parameter of the P-distance (e.g. 2).
%     - `layer.noRoot` it tells whether to raise the distance to
%     the P-th power (e.g. `false`).
%     - `layer.epsilon` is the regularization parameter for the derivatives.
%
%   Custom layer::
%     This can be used to specify custom layers.
%
%     - `layer.type` contains the string `'custom'`.
%     - `layer.forward` is  a function handle computing the block.
%     - `layer.backward` is a function handle computing the block derivative.
%
%     The first function is called as
%
%          res(i+1) = layer.forward(layer, res(i), res(i+1))
%
%     where RES is the structure array specified before. The second function is
%     called as
%
%          res(i) = layer.backward(layer, res(i), res(i+1))
%
%     Note that the `layer` structure can contain additional custom
%     fields if needed.
%
%   See also: dagnn.DagNN, VL_SIMPLENN_TIDY(),
%   VL_SIMPLENN_DISPLAY(), VL_SIMPLENN_MOVE().

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;
opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;
backPropLim = max(n - opts.backPropDepth + 1, 1);

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
  if opts.skipForward
    error('simplenn:skipForwardNoBackwPass', ...
      '`skipForward` valid only when backward pass is computed.');
  end
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

switch lower(opts.mode)
  case 'normal'
    testMode = false ;
  case 'test'
    testMode = true ;
  otherwise
    error('Unknown mode ''%s''.', opts. mode) ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  if opts.skipForward
    error('simplenn:skipForwardEmptyRes', ...
    'RES structure must be provided for `skipForward`.');
  end
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'stats', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end

if ~opts.skipForward
  res(1).x = x ;
end


% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
  if opts.skipForward, break; end;
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        l.opts{:}, ...
        cudnn{:}) ;
    case 'conv3' % FML addition
      res(i+1).x = mex_conv3d(res(i).x, l.weights{1}, l.weights{2}, ...
                              'pad', l.pad, ...
                              'stride', l.stride) ;

    case 'convt'
      res(i+1).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
        'crop', l.crop, ...
        'upsample', l.upsample, ...
        'numGroups', l.numGroups, ...
        l.opts{:}, ...
        cudnn{:}) ;

    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
        'pad', l.pad, 'stride', l.stride, ...
        'method', l.method, ...
        l.opts{:}, ...
        cudnn{:}) ;
    case 'pool3' % FML addition
      [res(i+1).x, res(i+1).aux] = mex_maxpool3d(...
          res(i).x, 'pool', l.pool, ...
          'pad', l.pad, 'stride', l.stride) ;

    case {'normalize', 'lrn'}
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;

    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;

    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;

    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;

    case 'sigentloss' % FML addition
      res(i+1).x = fml_nnsigentloss(res(i).x, l.class, l.target) ;
    case 'squaredloss' % FML addition
      res(i+1).x = fml_nnsquaredloss(res(i).x, l.class) ;
    case 'objloss' % FML addition
      res(i+1).x = fml_nncombobjloss(res(i).x, l.class, ...
                                     l.target, l.rnorm);

    case 'relu'
      if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
      res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;

    case 'sigmoid'
      res(i+1).x = vl_nnsigmoid(res(i).x) ;
    case 'objout' % FML addition
      res(i+1).x = fml_nnobjout(res(i).x);

    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;

    case 'spnorm'
      res(i+1).x = vl_nnspnorm(res(i).x, l.param) ;

    case 'dropout'
      if testMode
        res(i+1).x = res(i).x ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end

    case 'bnorm'
      if(~isfield(l, 'is3d') || ~l.is3d) % normal 2d data
      if testMode
        res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, 'moments', l.weights{3}) ;
      else
        res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}) ;
      end
      else % FML addition - 3d bnorm by collapsing 1st,2nd dims
        x_sz = size(res(i).x);
        if(length(x_sz)==4), x_sz(5) = 1; end
        x_in = reshape(res(i).x, [], x_sz(3), x_sz(4), x_sz(5));
        if testMode
          x_out = vl_nnbnorm(x_in, l.weights{1}, l.weights{2}, 'moments', l.weights{3}) ;
        else
          x_out = vl_nnbnorm(x_in, l.weights{1}, l.weights{2}) ;
        end
        res(i+1).x = reshape(x_out, x_sz);
      end

    case 'pdist'
      res(i+1).x = vl_nnpdist(res(i).x, l.class, l.p, ...
        'noRoot', l.noRoot, ...
        'epsilon', l.epsilon, ...
        'aggregate', l.aggregate) ;

    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;

    otherwise
      error('Unknown layer type ''%s''.', l.type) ;
  end

  % optionally forget intermediate results
  forget = opts.conserveMemory & ~(doder & n >= backPropLim) ;
  if i > 1
    lp = net.layers{i-1} ;
    % forget RELU input, even for BPROP
    forget = forget & (~doder | (strcmp(l.type, 'relu') & ~lp.precious)) ;
    forget = forget & ~(strcmp(lp.type, 'loss') || strcmp(lp.type, 'softmaxloss')) ;
    forget = forget & ~lp.precious ;
  end
  if forget
    res(i).x = [] ;
  end

  if gpuMode && opts.sync
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

% -------------------------------------------------------------------------
%                                                             Backward pass
% -------------------------------------------------------------------------

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:max(1, n-opts.backPropDepth+1)
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type

      case 'conv'
        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
          vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
          'pad', l.pad, ...
          'stride', l.stride, ...
          l.opts{:}, ...
          cudnn{:}) ;
      case 'conv3' % FML addition
        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
          mex_conv3d(res(i).x, l.weights{1}, l.weights{2}, ...
                     res(i+1).dzdx, ...
                     'pad', l.pad, 'stride', l.stride) ;

      case 'convt'
        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
          vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
          'crop', l.crop, ...
          'upsample', l.upsample, ...
          'numGroups', l.numGroups, ...
          l.opts{:}, ...
          cudnn{:}) ;

      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method, ...
                                l.opts{:}, ...
                                cudnn{:}) ;
      case 'pool3' % FML addition
        res(i).dzdx = mex_maxpool3d(...
            res(i+1).dzdx, res(i+1).aux, size(res(i).x), ...
            'pool', l.pool, ...
            'pad', l.pad, 'stride', l.stride);

      case {'normalize', 'lrn'}
        res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;

      case 'softmax'
        res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;

      case 'loss'
        res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;

      case 'softmaxloss'
        res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'sigentloss' % FML addition
        res(i).dzdx = fml_nnsigentloss(res(i).x, l.class, l.target, res(i+1).dzdx);
      case 'squaredloss' % FML addition
        res(i).dzdx = fml_nnsquaredloss(res(i).x, l.class, ...
                                        res(i+1).dzdx);
      case 'objloss' % FML addition
        res(i).dzdx = fml_nncombobjloss(res(i).x, l.class, ...
                                        l.target, l.rnorm, ...
                                        res(i+1).dzdx);

      case 'relu'
        if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
        if ~isempty(res(i).x)
          res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
        end

      case 'sigmoid'
        res(i).dzdx = vl_nnsigmoid(res(i).x, res(i+1).dzdx) ;

      case 'noffset'
        res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;

      case 'spnorm'
        res(i).dzdx = vl_nnspnorm(res(i).x, l.param, res(i+1).dzdx) ;

      case 'dropout'
        if testMode
          res(i).dzdx = res(i+1).dzdx ;
        else
          res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, ...
                                     'mask', res(i+1).aux) ;
        end

      case 'bnorm'
        if(~isfield(l, 'is3d') || ~l.is3d) % normal 2d data
        [res(i).dzdx, dzdw{1}, dzdw{2}, dzdw{3}] = ...
          vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx) ;
        % multiply the moments update by the number of images in the batch
        % this is required to make the update additive for subbatches
        % and will eventually be normalized away
        dzdw{3} = dzdw{3} * size(res(i).x,4) ;
        else % FML addition - 3d bnorm
          x_sz    = size(res(i).x);
          if(length(x_sz)==4), x_sz(5) = 1; end
          x_in    = reshape(res(i).x,      [], ...
                            x_sz(3), x_sz(4), x_sz(5));
          dzdx_in = reshape(res(i+1).dzdx, [], ...
                            x_sz(3), x_sz(4), x_sz(5));

          [dzdx_out, dzdw{1}, dzdw{2}, dzdw{3}] = ...
            vl_nnbnorm(x_in, l.weights{1}, l.weights{2}, dzdx_in) ;
          dzdw{3} = dzdw{3} * size(res(i).x,5) ;

          res(i).dzdx = reshape(dzdx_out, x_sz);
        end
      case 'pdist'
        res(i).dzdx = vl_nnpdist(res(i).x, l.class, ...
          l.p, res(i+1).dzdx, ...
          'noRoot', l.noRoot, ...
          'epsilon', l.epsilon, ...
          'aggregate', l.aggregate) ;

      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;

    end % layers

    switch l.type % FML addition
      case {'conv', 'conv3', 'convt', 'bnorm'}
        if ~opts.accumulate
          res(i).dzdw = dzdw ;
        else
          for j=1:numel(dzdw)
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
        end
        dzdw = [] ;
    end
    if opts.conserveMemory && ~net.layers{i}.precious && i ~= n
      res(i+1).dzdx = [] ;
      res(i+1).x = [] ;
    end
    if gpuMode && opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end
