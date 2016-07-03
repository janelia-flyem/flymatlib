function [pred_labels, acc, gt_labels] = ...
      mlp_test(this, feat_output_fns, num_points, ...
               table_name, gm, gs, edge_id, ...
               ids, start_offset, num_process)

  use_db = 0;
  if(~exist('start_offset','var') || isempty(start_offset))
    start_offset = 0;
  end
  if(~exist('num_process','var') || isempty(num_process))
    num_process = num_points;
  end
  
  if(isscalar(feat_output_fns))
    use_db = 1;
    % num_features = feat_output_fns;
    gt_labels = zeros(num_process, 1);
    
    assert(exist('table_name','var') > 0 && ...
           ~isempty(table_name), ...
           'JANLIB:AssertionFailed', ...
           'table name not provided');
  else
    if(~iscell(feat_output_fns))
      test_feats   = feat_output_fns;
      % num_features = size(test_feats, 2);
    else
      dc_index = 0;
      for i=1:length(feat_output_fns)
        dc = load(feat_output_fns{i});
        if(dc_index == 0)
          test_feats  = zeros(num_process, size(dc.feats,2), ...
                              'single');
          test_labels = zeros(num_process, 1);
        end
        
        dc_n = size(dc.feats,1);
        test_feats(dc_index+1:dc_index+dc_n, :) = dc.feats;
        test_labels(dc_index+1:dc_index+dc_n) = dc.labels;
        dc_index = dc_index + dc_n;
      end
      gt_labels = test_labels;
      % num_features = size(test_feats, 2);
    end
  end
    
  max_batchsize = 1000;
  pred_labels   = zeros(num_process, 1);
  num_instances = num_points;
  dropout_rate_input_copy  = this.dropout_rate_input;
  dropout_rate_hidden_copy = this.dropout_rate_hidden;
  if(isscalar(dropout_rate_hidden_copy))
    dropout_rate_hidden_copy = ...
        dropout_rate_hidden_copy * ones(size(this.num_hidden));
  end
  
  dropout_correction_input = 1 - dropout_rate_input_copy;

  if(use_db == 1)
    if(~exist('ids','var') || isempty(ids))
      [~,~, ids] = ka_db_retrieve(table_name, 1, 1, [], edge_id);
    end
  end
  
  for i=(start_offset+1):max_batchsize:(start_offset+num_process)
    i_ulim = min(i+max_batchsize-1, ...
                 min(start_offset+num_process, num_instances));
    
    if(use_db == 1)
      [y0, y0_labels] = ka_db_retrieve(table_name, ...
                                       i_ulim-i+1, i, ids, edge_id);
      % normalize
      y0 = bsxfun(@rdivide, ...
                  bsxfun(@minus, y0, gm), gs);
      
      gt_labels((i-start_offset):(i_ulim-start_offset)) = ...
          y0_labels;
      y0 = y0';
    else
      y0 = test_feats((i-start_offset):(i_ulim-start_offset),:)';
    end
    
    y0 = y0 * dropout_correction_input;

    [~,pred_labels((i-start_offset):(i_ulim-start_offset))] = ...
        thresh_linear_sigmoid(y0, this.w1, this.b1, ...
                              this.w2, this.b2, ...
                              -dropout_rate_hidden_copy);
  end
  
  if(nargout > 1)
    pl_bin = pred_labels > 0.5;
    tl_bin = gt_labels > 0;
    
    acc = mean(~xor(pl_bin, tl_bin));
  end
end
