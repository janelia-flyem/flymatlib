base_dir = fileparts(mfilename('fullpath'));

for ii=1:2
  fns_json{ii} = sprintf('%s/ex_cube%d.json', base_dir, ii);
  fns_im{ii}   = sprintf('%s/ex_cube%d.h5',   base_dir, ii);
end

% generate labels/mask h5 files
for ii=1:2
  tbar_json2labelmask(fns_json{ii}, ...
                      sprintf('%s/ex_cube%d_', base_dir, ii), ...
                      [250 250 250], 7, 5);
end

% train a simple CNN classifier
%   using a fairly simple network model and only training for
%   a few epochs, this will take less than half an hour on a CPU
%   being able to train on a GPU is highly recommended in order
%   to train more complex networks
opts.expDir              = sprintf('%s/fml_example', base_dir);
opts.imdbPath            = { ...
    fns_im{1}, sprintf('%s/ex_cube1_', base_dir); ...
    fns_im{2}, sprintf('%s/ex_cube2_', base_dir) };
opts.learningRate        = [0.01*ones(1,5)];
opts.nums                = [3e4, 3e4];
opts.ratios              = [0.5 0.5];
opts.classes             = [0 1];

opts.weightDecay         = 0.0001;
opts.batchSize           = 100;
opts.gpus                = [];
opts.is_autoencoder      = false;
opts.data_aug            = true;
opts.fraction_to_corrupt = 0;

model_str = ...
    'c3x48-b-r-m3x2-c3x48-b-r-m3x2-c1x96-b-r-c1x1-e1';
net = tbar_cnn_init(model_str);
fml_simplenn_display(net,'inputSize', [15 15 1 100])
[net, info] = tbar_cnn_train(net, opts);

% run inference of training/validation volumes
% also compute precision/recall
dist_thresh  = 30;
thds         = 0.5:0.01:0.95;
ave_radius   = 7;
obj_min_dist = 24;
remove_bf    = 5;
vol_sz       = [250 250 250];

epoch       = 5;
net_save = load(sprintf('%s/net-epoch-%d.mat', opts.expDir, epoch));
net      = tbar_cnn_finalize_net(net_save.net);

for ii=1:2
  [num_tp, tot_pred, tot_gt] = ...
      tbar_cnn_test(net, opts.expDir, epoch, ...
                    fns_im{ii}, fns_json{ii}, 1, ...
                    dist_thresh, thds, ave_radius, obj_min_dist, ...
                    remove_bf, vol_sz);
  pp{ii} = num_tp ./ tot_pred;
  rr{ii} = num_tp ./ tot_gt;
end

figure, hold on
cc = {'r-','b-'};
for ii=1:2
  plot(rr{ii},pp{ii},cc{ii});
end
axis([0.5 1 0.5 1]);
grid on
legend('training', 'validation', 'Location', 'SouthWest');
