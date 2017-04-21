function pp = psd_make_global_unique(pp, dvid_conn, seg_name)

  if(isempty(pp)), return, end

  vol_of     = [3 3 3];
  vol_sz     = 2*vol_of - 1;
  [xx,yy,zz] = ndgrid(1:vol_sz(1),1:vol_sz(2),1:vol_sz(3));
  xx = xx(:); yy = yy(:); zz = zz(:);
  n_of = length(xx);

  ppi = pp;
  for ii=1:length(ppi)
    if(isempty(pp{ii}))
      pp{ ii} = zeros(4,0);
      ppi{ii} = pp{ii};
    end
    ppi{ii}(5,:) = ii;
  end
  pa = cell2mat(ppi);
  [~,unq_idx]      = unique(pa(1:3,:)','rows');
  dup_idx          = true(1,size(pa,2));
  dup_idx(unq_idx) = false;
  dup_syn          = pa(5,dup_idx);
  n_dups_tot       = length(dup_syn);

  syn_dups_idx = unique(dup_syn);
  syn_dups     = false(1,length(ppi));
  syn_dups(syn_dups_idx) = true;
  syn_unqs_idx = find(~syn_dups);
  n_dups = 0;
  for dd=1:length(syn_dups_idx)
    ii = syn_dups_idx(dd);
    if(isempty(pp{ii})), continue, end

    locs_prev = cell2mat(pp(...
        [syn_unqs_idx, syn_dups_idx(1:(dd-1))]));
    locs_prev = locs_prev(1:3,:)';

    dup_vec = ismember(pp{ii}(1:3,:)',locs_prev,'rows');

    n_dups = n_dups + sum(dup_vec);

    idx = 1:size(pp{ii},2);
    for jj=idx(dup_vec)
      % pull down 5^3 window centered as psd
      seg_fn = sprintf(...
          'tmp_pb_psd_%d-%d-%d-%s.h5', ...
          pp{ii}(1:3,jj)', datestr(now,30));
      dvid_conn.get_segmentation(...
          pp{ii}(1:3,jj)' - vol_of, vol_sz, seg_fn, seg_name);
      ss = read_image_stack(seg_fn);
      delete(seg_fn);

      % find loc that is in same seg and unique
      got_match = false;
      for kk=1:n_of
        if( ss( yy(kk),xx(kk),zz(kk) ) ~= ...
            ss(vol_of(2)+1,vol_of(1)+1,vol_of(3)+1) )
          continue
        end
        nx = pp{ii}(1,jj) + xx(kk) - vol_of(1);
        ny = pp{ii}(2,jj) + yy(kk) - vol_of(2);
        nz = pp{ii}(3,jj) + zz(kk) - vol_of(3);
        if( ismember([nx ny nz],locs_prev,'rows') )
          continue
        end

        got_match = true;
        pp{ii}(1:3,jj) = [nx ny nz]';
        break
      end

      if(~got_match), keyboard, end
    end
  end
  if(n_dups ~= n_dups_tot),
    disp([n_dups n_dups_tot]);
  end
  assert(n_dups >= n_dups_tot,'FML:AssertionFailed',...
         'unexpected number of duplicates');
end
