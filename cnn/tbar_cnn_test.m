function [num_tp, tot_pred, tot_gt, values_raw, values_conf] = ...
  tbar_cnn_test(net, exp_dir, epoch, image_fn, test_json, ...
                n_workers, ...
                dist_thresh, thds, ave_radius, obj_min_dist, ...
                remove_buffer_radius, vol_sz, ...
                obj_ave_radius, values_raw, values_conf)
% TBAR_CNN_TEST tbar cnn precision/recall
% [num_tp, tot_pred, tot_gt] = ...
%   TBAR_CNN_TEST(net, exp_dir, epoch, image_fn, test_json, ...
%                 dist_thresh, thds, ave_radius, obj_min_dist, ...
%                 remove_buffer_radius, vol_sz, ...
%                 n_workers)
%
%   net         matconvnet network
%   exp_dir     directory to save output files
%   epoch       network epoch (check if output file already exists)
%   image_fn    image filename
%   gt_json     groundtruth json filename
%   n_workers   number of distributed workers to use
%
%   dist_thresh, thds, ave_radius, obj_min_dist, ...
%   remove_buffer_radius, vol_sz   see tbar_pr_curve
%
%   obj_ave_radius, values_raw, values_conf
%      see tbar_collapse_objout

  if(~exist(exp_dir,'dir'))
    system(sprintf('mkdir -p %s', exp_dir));
  end
  if(~exist('remove_buffer_radius','var'))
    remove_buffer_radius = [];
  end
  if(~exist('vol_sz','var'))
    vol_sz = [];
  end
  if(~exist('obj_ave_radius','var'))
    obj_ave_radius = [];
  end
  if(~exist('values_raw','var'))
    values_raw = [];
  end
  if(~exist('values_conf','var'))
    values_conf = [];
  end

  if(strcmp(net.layers{end}.type, 'objout'))
    is_objout = true;
  else
    is_objout = false;
  end

  if(~isempty(epoch))
    suffix = sprintf('_ep%03d', epoch);
  else
    suffix = '';
  end
  [~,prefix,~] = fileparts(image_fn);
  out_fn   = sprintf('%s/%s_out%s.h5',   exp_dir, prefix, suffix);
  out_json = sprintf('%s/%s_out%s.json', exp_dir, prefix, suffix);
  thd_json = sprintf('%s/%s_out%s_md%g_ar%g.json', ...
                     exp_dir, prefix, suffix, ...
                     obj_min_dist, ave_radius);
  if(is_objout)
    out_fn_c = sprintf('%s/%s_out_c%s.h5', exp_dir, prefix, suffix);
  else
    out_fn_c = out_fn;
  end

  if(isempty(epoch) || ~exist(out_fn_c,'file'))
    tbar_cnn_infer(net, image_fn, out_fn_c, n_workers);
  end
  if(is_objout)
    [values_raw, values_conf] = ...
        tbar_collapse_objout(out_fn_c, out_fn, [], ...
                             obj_ave_radius, ...
                             values_raw, values_conf);
  end
  if(isempty(epoch) || ~exist(thd_json,'file'))
    tbar_voxel2obj(out_fn, out_json, thds(1), ...
                   ave_radius, obj_min_dist);
  end

  pr_out = sprintf('%s/%s_out%s_thds%g-%g_dt%g_md%g_ar%g.mat', ...
                   exp_dir, prefix, suffix, ...
                   thds(1), thds(end), dist_thresh, ...
                   obj_min_dist, ave_radius);
  if(isempty(epoch) || ~exist(pr_out,'file'))
    [~,~, num_tp, tot_pred, tot_gt] = ...
        tbar_pr_curve(out_json, test_json, dist_thresh, thds, ...
                      remove_buffer_radius, vol_sz);
    save(pr_out,'num_tp','tot_pred','tot_gt');
  else
    ss = load(pr_out,'num_tp','tot_pred','tot_gt');
    num_tp = ss.num_tp; tot_pred = ss.tot_pred; tot_gt = ss.tot_gt;
  end
end
