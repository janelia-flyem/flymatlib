function mlp_code_test()

  num_instances  = 10000;
  num_features   = 1;
  num_hidden     = 100;
  minibatch_size = 10;
  dropout_rate   = 0;
  
  num_rows      = 1;
  indices       = ceil(num_features*rand(1,num_rows));

  train_feats   = 1*randn(num_instances, num_features);
  train_labels  = rand(num_instances,1) > 0.5;
  test_feats    = 1*randn(num_instances, num_features);
  test_labels   = rand(num_instances,1) > 0.5;

  train_feats(:, indices) = 2*repmat(train_labels, 1, num_rows)-1;
  test_feats(:, indices)  = 2*repmat(test_labels, 1, num_rows)-1;

  mo = mlp(num_hidden);
  mo.dropout_rate_hidden = dropout_rate;
  tic
  mo.mlp_train(train_feats, train_labels, [], 10);
  [~,acc] = mo.mlp_test(test_feats, test_labels);
  acc
  toc
  figure
  tic
  [w1,b1,w2,b2,meanerr,pcntg] = ...
      mlp_minibatch_dropout_outgoingfix_threshlinear_vec3_sqsq(...
        train_feats', train_labels', num_hidden, 3*num_instances, ...
        test_feats', test_labels', 0, 0, ...
        minibatch_size, 15, dropout_rate, 0.3);
  toc
end

