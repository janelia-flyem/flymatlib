function [train_feats, train_labels, train_data, got_all] = ...
    get_training_batch(this, num_subinstances, ...
                             edge_ids, pos_ratio_copy, ...
                             train_data)

  train_feats  = this.feats;  % n_obj x feat_dim
  train_labels = this.labels; % n_obj x label_dim
  train_data   = [];
  got_all      = 1;
  
  % currently this assumes that pos_ratio_copy = -1,
  % otherwise errors will occur
end
