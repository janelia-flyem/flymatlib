function remove_tbar = tbar_global_apply_mindist(locs, min_dist)

  remove_tbar = zeros(1,size(locs,2));

  [idx,d] = knnsearch(locs(1:3,:)',locs(1:3,:)','K',2);
  not_valid = d(:,2) < min_dist;
  if(sum(not_valid) == 0)
    fprintf('orig set all valid\n');
    return
  end
  
  locs_kdt = KDTreeSearcher(locs(1:3,:)');

  not_valid( idx(not_valid,2) ) = 1;
  [~,idx] = sort(-locs(4,:) .* not_valid');
  idx     = idx(1:sum(not_valid));
  
  fprintf('%d\n', length(idx));

  to_remove = rangesearch(locs_kdt, locs(1:3, idx)', min_dist);
  
  for ii = 1:length(idx)
    if(remove_tbar(idx(ii))), continue, end    
    remove_tbar(to_remove{ii}) = 1;
    remove_tbar(idx(ii)) = 0;
  end
end
