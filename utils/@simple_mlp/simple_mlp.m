classdef simple_mlp < handle

  properties
    threaded = 0
    
    name
    mlp_init
    model
    
    feats
    labels
  end
  
  methods
    function obj = simple_mlp(name, mlp_init)
      obj.name     = name;
      obj.mlp_init = mlp_init;
    end

    function p = do_inference(this, feature_vec)
      p = 2*this.model.mlp_test(feature_vec, ...
                                size(feature_vec,1)) - 1;
    end

    % defined externally
    do_training(this, feats, labels)
    [train_feats, train_labels, train_data, got_all] = ...
        get_training_batch(this, num_subinstances, ...
                                 edge_ids, pos_ratio_copy, ...
                                 train_data)
  end
end
