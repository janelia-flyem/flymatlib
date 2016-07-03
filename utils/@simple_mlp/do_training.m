function do_training(this, feats, labels)

  global DFEVAL_DIR
  
  this.feats  = feats;
  this.labels = labels;
  
  n_features = size(feats,2);
  n_examples = size(feats,1);
  
  the_global_mean = zeros(1,n_features);
  the_global_std  = ones( 1,n_features);
  
  this.model = this.mlp_init.copy();
  this.model.mlp_train(...
    n_features, n_examples, this.name, ...
    the_global_mean, the_global_std, 1, ...
    DFEVAL_DIR, [], this);
  
  this.feats  = [];
  this.labels = [];
  
  % TODO: for now, this is just running locally
  %       should make this a distributed call to allow
  %       running on gpu
end
