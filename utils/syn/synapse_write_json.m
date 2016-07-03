function [tlocs, plocs] = ...
        synapse_write_json(fn, locs, pred, z_offset, thresholds, ...
                           tbar_conf)

  if(~exist('z_offset','var') || isempty(z_offset))
    z_offset = 0;
  end

  for thresh = thresholds
    n_synapses = size(locs,1);
    tlocs      = ones(4,n_synapses);
    plocs      = cell(1,n_synapses);
    
    tt  = 1;
    idx = 0;
    tc_idx = 1;
    while(tt <= n_synapses)
      idx = idx + 1;

      tbar           = locs(tt,1:3);
      tlocs(1:3,idx) = tbar';
      if(exist('tbar_conf','var') && ~isempty(tbar_conf))
          while(~isequal(tlocs(1:3,idx), ...
                         tbar_conf(1:3,tc_idx)))
              tc_idx = tc_idx + 1;
          end
          tlocs(4,idx) = tbar_conf(4, tc_idx);
      end
      
      tt_end = tt;
      while(tt_end < n_synapses && ...
            isequal(locs(tt_end+1,1:3), tbar))
	tt_end = tt_end+1;
      end
      
      n_psds     = tt_end - tt + 1;
      plocs{idx} = zeros(4, n_psds);

      psd_idx = 0;
      for jj = 1:n_psds
	tt_iter = tt+jj-1;

	if(pred(tt_iter) > thresh)
	  psd_idx = psd_idx + 1;
	  plocs{idx}(1:3,psd_idx) = locs(tt_iter,4:6)';
	  plocs{idx}(4,  psd_idx) = pred(tt_iter);
	end
      end

      plocs{idx}      = plocs{idx}(:,1:psd_idx);
      plocs{idx}(3,:) = plocs{idx}(3,:) + z_offset;
      
      tt  = tt_end + 1;
    end
    
    tlocs = tlocs(:,1:idx);
    plocs = plocs(:,1:idx);
    
    tlocs(3,:) = tlocs(3,:) + z_offset;

    if(exist('fn','var') && ~isempty(fn))
        fn_thresh = sprintf('%s_%g.json', fn, thresh);
    
        tbar_psd_json_write(fn_thresh, tlocs, plocs);
    end
  end
  
end
