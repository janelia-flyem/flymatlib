classdef mlp < handle

  properties
    use_gpu                = 1
    pos_ratio              = 0.5 % only works with db
    
    eta_w_start            = [.1 .1]
    eta_b_start            = [.1 .1]
    eps_init               = 0.01
    
    dampening_factor       = 1.0; % old value 0.95
    lambda_l2              = 0
    
    dropout_rate_input     = 0
    dropout_rate_hidden    = 0.5
    w1_mag_thresh          = 15
    margin                 = 0.3
    inv_margin             = 0
    
    use_adagrad            = 0
    
    loss_func              = 0    % 0 = squared error
                                  % 1 = cross-entropy
    
    num_updates_default    = 2e5;
    minibatch_size_default = 40;
  end
  
  properties (GetAccess = public, SetAccess = private)
    num_hidden
    w1
    b1
    w2
    b2
    
    training_errsq
    training_classerr
  end
  
  methods % defined externally
    mlp_train(this, num_features, num_points, ...
              table_name, gm, gs, edge_id, dfeval_dir, ...
              max_job_id, ec_mlp_obj)
    [pred_labels, acc, gt_labels] = ...
        mlp_test(this, feat_output_fns, num_points, ...
                 table_name, gm, gs, edge_id, ...
                 ids_orig, start_offset, num_process)
  end
  
  methods
    function obj = mlp(num_hidden)
      if(nargin > 0)
        obj.num_hidden = num_hidden;
      end
    end
    
    function [save_errsq, save_classerr] = clear_histories(this)
      save_errsq    = this.training_errsq;
      save_classerr = this.training_classerr;
      
      this.training_errsq    = [];
      this.training_classerr = [];
    end
    
    function restore_histories(this, save_errsq, save_classerr)
      this.training_errsq    = save_errsq;
      this.training_classerr = save_classerr;
    end
    
    function copy_weights(this, m, n_layers)
      if(~iscell(m.w1))
        m.w1 = { m.w1 };
      end
      if(~exist('n_layers','var') || isempty(n_layers))
        n_layers = length(m.w1);
      end
      
      nhu_nlayers = length(this.num_hidden);
      this.w1 = cell(1, nhu_nlayers);
      this.b1 = cell(1, nhu_nlayers);
      
      for ii=1:n_layers
        nhu   = this.num_hidden(ii);
        w_nhu = size(m.w1{ii},2); 
        assert(nhu == w_nhu, ...
               'JANCOM:AssertionFailed', ...
               sprintf('incompatible weight size: %d ~= %d', ...
                       nhu, w_nhu));
        this.w1{ii} = m.w1{ii};
        this.b1{ii} = m.b1{ii};
      end
    end
    
    function join(this, varargin)
    % initialize weights for joint mlp classifier from
    % individual classifiers
      n_mlps = length(varargin);
      
      this.w1 = varargin{1}.w1;
      this.b1 = varargin{1}.b1;
      
      this.w2 = zeros(size(varargin{1}.w2,1), ...
                      n_mlps);
      this.b2 = zeros(n_mlps, 1);
      
      for ii=1:n_mlps
        this.w2(:, ii) = varargin{ii}.w2;
        this.b2(ii)    = varargin{ii}.b2;
      end
    end
    
    function obj = split(this, edge_ids)
      n_mlps = size(this.w2,2);
      if(~exist('edge_ids','var') || isempty(edge_ids))
        edge_ids = 1:n_mlps;
      else
        assert(length(edge_ids) == n_mlps, ...
               'JANCOM:AssertionFailed', ...
               sprintf( ...
                 '%d == length(edge_ids) ~= n_mlps == %d', ...
                 length(edge_ids), n_mlps));
      end
      max_edge_id = max(edge_ids);
      obj         = cell(1, max_edge_id);
      
      for ii=1:n_mlps
        obj{edge_ids(ii)}    = copy(this);
        obj{edge_ids(ii)}.w2 = this.w2(:,ii);
        obj{edge_ids(ii)}.b2 = this.b2(ii);
      end
    end
    
    function obj = copy(this)
      obj = mlp(this.num_hidden);
      
      obj.use_gpu     = this.use_gpu;
      obj.pos_ratio   = this.pos_ratio;
      
      obj.eta_w_start = this.eta_w_start;
      obj.eta_b_start = this.eta_b_start;
      obj.eps_init    = this.eps_init;
      
      obj.dampening_factor = this.dampening_factor;
      obj.lambda_l2        = this.lambda_l2;
      obj.loss_func        = this.loss_func;
      
      obj.dropout_rate_input  = this.dropout_rate_input;
      obj.dropout_rate_hidden = this.dropout_rate_hidden;
      obj.w1_mag_thresh       = this.w1_mag_thresh;
      obj.margin              = this.margin;
      obj.inv_margin          = this.inv_margin;
      obj.use_adagrad         = this.use_adagrad;
      
      obj.num_updates_default    = this.num_updates_default;
      obj.minibatch_size_default = this.minibatch_size_default;
      
      obj.w1 = this.w1;
      obj.b1 = this.b1;
      obj.w2 = this.w2;
      obj.b2 = this.b2;
      
      obj.training_errsq    = this.training_errsq;
      obj.training_classerr = this.training_classerr;
    end
  end
end
