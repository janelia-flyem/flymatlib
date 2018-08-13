%%% script demonstrating psd training using FIB25 data

% set directory for writing data/model
base_dir = 'fib25_psd_save'


%% training details, source
vol_start = [2508, 2759, 1500; ...
             3969, 3732, 2500];
vol_sz = [520, 520, 520];
vol_bf = 30;

n_cubes = size(vol_start,1);

dvid_conn = dvid_connection(...
    'emdata.janelia.org', '8225', 'fml_user', 'fml');

% download grayscale, segmentation
fprintf('preparing grayscale, segmentation ...\n');
do_normalize = [128, 33];
for ii = 1:n_cubes
  image_fn = sprintf('%s/cube%02d_img.h5', base_dir, ii);
  dvid_conn.get_image(...
      vol_start(ii,:), vol_sz, image_fn, do_normalize);
  segm_fn = sprintf('%s/cube%02d_sgm.h5', base_dir, ii);
  dvid_conn.get_segmentation(...
      vol_start(ii,:), vol_sz, segm_fn, 'groundtruth');
end

% download synapses
% from original source
fprintf('preparing synapses ...\n');
% syn_url = ...
%   'http://emdata.janelia.org/api/node/8225/.files/key/synapse.json';
% syn_all_fn = sprintf('%s/syn_all.json', base_dir);
% websave(syn_all_fn, syn_url);
% [tbars, psds] = tbar_json2locs(syn_all_fn, [], true);
% load from saved subset
load('fib25_syn_gt.mat');
for ii = 1:n_cubes
  % write out cube synapses in local Raveler coordinates (flip y)
  tt_idx = ...
      tbars(1,:) >= vol_start(ii,1) + vol_bf & ...
      tbars(2,:) >= vol_start(ii,2) + vol_bf & ...
      tbars(3,:) >= vol_start(ii,3) + vol_bf & ...
      tbars(1,:) < vol_start(ii,1) + vol_sz(1) - vol_bf & ...
      tbars(2,:) < vol_start(ii,2) + vol_sz(2) - vol_bf & ...
      tbars(3,:) < vol_start(ii,3) + vol_sz(3) - vol_bf;

  tbars_cube = tbars(:, tt_idx);
  psds_cube = psds(tt_idx);

  tbars_cube(1:3,:) = ...
      bsxfun(@minus, tbars_cube(1:3,:), vol_start(ii,:)');
  tbars_cube(2,:)   = vol_sz(2) - tbars_cube(2,:) - 1;
  for jj=1:size(tbars_cube,2)
    if(isempty(psds_cube{jj})), continue, end
    psds_cube{jj}(1:3,:) = bsxfun(...
        @minus, psds_cube{jj}(1:3,:), vol_start(ii,:)');
    psds_cube{jj}(2,:)   = vol_sz(2) - psds_cube{jj}(2,:) - 1;
  end

  syn_fn = sprintf('%s/cube%02d_syn.json', base_dir, ii);
  tbar_psd_json_write(syn_fn, tbars_cube, psds_cube);
end


%% extract features/training examples
all_seg_candidates = true;
use_v2 = true;

for ii=1:n_cubes
  json_gt{ii} = sprintf('%s/cube%02d_syn.json', base_dir, ii);
  seg_fn{ii} = sprintf('%s/cube%02d_sgm.h5', base_dir, ii);
  im_fn{ii} = sprintf('%s/cube%02d_img.h5', base_dir, ii);
end

% feature params
window_radii = [18 26 40];
dilate_radii = [3  5  7];
image_thresh = [-0.7 -0.9 -1.1 -1.3];

neighbor_thresh = 2;

ll   = cell(1,n_cubes);
ff   = cell(1,n_cubes);
syn  = cell(1,n_cubes);
locs = cell(1,n_cubes);

for ii=1:length(json_gt)
  [ll{ii}, ff{ii}, syn{ii}, locs{ii}] = synapse_labels_feats(...
    json_gt{ii}, seg_fn{ii}, im_fn{ii}, [], ...
    15, [520 520 520], ...
    window_radii, dilate_radii, image_thresh, [], ...
    use_v2,[],[],neighbor_thresh, all_seg_candidates);
end

train_idx = 1:n_cubes;

ll_train = cell2mat(ll(train_idx)');
ff_train = cell2mat(ff(train_idx)');

fmn      = mean(ff_train,1);
ff_train = bsxfun(@minus, ff_train, fmn);
fstd     = std(ff_train, [], 1);
ff_train = bsxfun(@rdivide, ff_train, fstd+1e-4);

%% train mlp
% mlp parameters
tic
mlp_init = mlp(25);
mlp_init.num_updates_default = 1e5;
mlp_init.minibatch_size_default = 40;
mlp_init.use_gpu = 0;
mlp_init.pos_ratio = -1;%0.5;
mlp_init.inv_margin = 0.1;
mlp_init.margin = 0.4;
mlp_init.eta_w_start = [0.02 0.02];
mlp_init.eta_b_start = [0.02 0.02];
mlp_init.loss_func = 1

mm = simple_mlp('fib25_mlp', mlp_init);
mm.do_training(ff_train, ll_train);
pp{1} = mm.do_inference(ff_train);
toc

% training performance
[~,~,~,auc_pr(1),prec_ind_mlp(1),recall_ind_mlp(1)] = ...
  compute_auc(ll_train,pp{1});
% to plot:
% plot(recall_ind_mlp{1},prec_ind_mlp{1},'b-')


%% save
psdm.psd_model          = mm;
psdm.do_normalize       = [128 33];
psdm.window_radii       = window_radii;
psdm.dilate_radii       = dilate_radii;
psdm.image_thresh       = image_thresh;
psdm.total_window       = [];
psdm.neighboring_dilate = neighbor_thresh;
psdm.psd_thresh         = 0.2;
psdm.ff_thresh          = [];%ff_thresh;
psdm.fmn                = fmn;
psdm.fstd               = fstd;
save(sprintf('%s/psd_trained.mat', base_dir), ...
     '-struct', 'psdm');
