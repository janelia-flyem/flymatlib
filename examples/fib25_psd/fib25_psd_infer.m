% directory for trained model and inference outputs
base_dir = 'fib25_psd_save'


machine_name = 'emdata.janelia.org';
repo_name    = '8225';
seg_name     = 'groundtruth';

dvid_conn = dvid_connection(...
    machine_name, repo_name, 'fml_user', 'fml');

fn_vol_start = 'fib25_psd_infer_roi.txt';

% syn_all_fn = sprintf('%s/syn_all.json', base_dir);
% [tbars, psds] = tbar_json2locs(syn_all_fn, [], true);
load('fib25_syn_gt.mat');

psd_model_fn = sprintf('%s/psd_trained2.mat', base_dir);
work_dir = sprintf('%s/output2', base_dir);

psd_full_infer(work_dir, psd_model_fn, tbars, ...
               fn_vol_start, [], ...
               dvid_conn, seg_name);


%% evaluate
fid = fopen(fn_vol_start);
vol_start = fscanf(fid, '%d', [6,Inf])';
vol_sz    = vol_start(:,4:6);
vol_start = vol_start(:,1:3);
fclose(fid);
n_cubes = size(vol_start,1);

for ii = 1:n_cubes
  tt_idx = ...
      tbars(1,:) >= vol_start(ii,1) & ...
      tbars(2,:) >= vol_start(ii,2) & ...
      tbars(3,:) >= vol_start(ii,3) & ...
      tbars(1,:) < vol_start(ii,1) + vol_sz(ii,1) & ...
      tbars(2,:) < vol_start(ii,2) + vol_sz(ii,2) & ...
      tbars(3,:) < vol_start(ii,3) + vol_sz(ii,1);

  tbars_sub{ii} = tbars(:, tt_idx);
  psds_sub{ii} = psds(tt_idx);
end

tbars_gt = cell2mat(tbars_sub);
psds_gt = horzcat(psds_sub{:});

ss = load(sprintf('%s/output2/syn.mat', base_dir));
tbars_pd = ss.tlocs;
psds_pd = ss.plocs;

[llt_gt, llp_gt] = dvid_conn.get_labels(...
    'groundtruth', tbars_gt, psds_gt);
[llt_pd, llp_pd] = dvid_conn.get_labels(...
    'groundtruth', tbars_pd, psds_pd);

[mm, bb] = synapse_pr_curve(llt_gt, llp_gt, llt_pd, llp_pd, ...
                            psds_pd, 0.2:0.1:0.9, ...
                            'groundtruth');
