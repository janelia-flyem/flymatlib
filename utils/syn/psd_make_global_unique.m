function [pp,bad_idx] = psd_make_global_unique(...
    pp, dvid_conn, seg_name, cube_seg_fn, seg_offset)

  bad_idx = zeros(0,2);
  if(isempty(pp)), return, end

  use_dvid = true;
  if(isempty(dvid_conn))
    use_dvid = false;
  end

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

  if(length(syn_dups_idx)>0 && ~use_dvid)
    local_seg = read_image_stack(cube_seg_fn);
  end
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
      if(use_dvid)
        % pull down 5^3 window centered as psd
        seg_fn = sprintf(...
            'tmp_pb_psd_%d-%d-%d-%s.h5', ...
            pp{ii}(1:3,jj)', datestr(now,30));
        dvid_conn.get_segmentation(...
            pp{ii}(1:3,jj)' - vol_of, vol_sz, seg_fn, seg_name, ...
            false, true, false);
        ss = read_image_stack(seg_fn);
        delete(seg_fn);
      else
        xxi = pp{ii}(2,jj) - vol_of(2) - seg_offset(2);
        yyi = pp{ii}(1,jj) - vol_of(1) - seg_offset(1);
        zzi = pp{ii}(3,jj) - vol_of(3) - seg_offset(3);

        if(xxi(1)<=0 || yyi(1)<=0 || zzi(1)<=0)
          continue
        end
        if(xxi(end)+vol_sz(1)>=size(local_seg,1) || ...
           yyi(end)+vol_sz(2)>=size(local_seg,2) || ...
           zzi(end)+vol_sz(3)>=size(local_seg,3))
          continue
        end
        ss = local_seg(xxi+(1:vol_sz(1)),...
                       yyi+(1:vol_sz(2)),...
                       zzi+(1:vol_sz(3)));
      end

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

      if(~got_match)
        fprintf('could not find possible match\n')
        bad_idx(end+1,:) = [ii,jj];
        if(exist('cube_seg_fn','var') && ...
           ~isempty(cube_seg_fn))
          fprintf('\t%s\n', cube_seg_fn)
        end
      end
    end
  end
  if(n_dups ~= n_dups_tot)
    disp(seg_name);
    disp([n_dups n_dups_tot]);
  end
  assert(n_dups >= n_dups_tot,'FML:AssertionFailed',...
         'unexpected number of duplicates');
end
