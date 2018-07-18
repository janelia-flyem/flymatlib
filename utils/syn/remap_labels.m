function [gt_labels_tt, gt_labels_pp, ...
          pd_labels_tt, pd_labels_pp] = remap_labels(...
              gt_labels_tt, gt_labels_pp, ...
              pd_labels_tt, pd_labels_pp)

  all_labels = [gt_labels_tt, cell2mat(gt_labels_pp), ...
                pd_labels_tt, cell2mat(pd_labels_pp)];
  [~,~,mapped_labels] = unique(all_labels);

  n_p = 0;

  n_n = length(gt_labels_tt);
  gt_labels_tt(:) = mapped_labels(n_p + (1:n_n));
  n_p = n_p + n_n;
  for ii = 1:size(gt_labels_pp,2)
    n_n = length(gt_labels_pp{ii});
    gt_labels_pp{ii}(:) = mapped_labels(n_p + (1:n_n));
    n_p = n_p + n_n;
  end

  n_n = length(pd_labels_tt);
  pd_labels_tt(:) = mapped_labels(n_p + (1:n_n));
  n_p = n_p + n_n;
  for ii = 1:size(pd_labels_pp,2)
    n_n = length(pd_labels_pp{ii});
    pd_labels_pp{ii}(:) = mapped_labels(n_p + (1:n_n));
    n_p = n_p + n_n;
  end

end
