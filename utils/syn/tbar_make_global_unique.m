function tt = tbar_make_global_unique(tt, pp, dvid_conn, seg_name)

  vol_of     = [3 3 3];
  vol_sz     = 2*vol_of - 1;
  [xx,yy,zz] = ndgrid(1:vol_sz(1),1:vol_sz(2),1:vol_sz(3));
  xx = xx(:); yy = yy(:); zz = zz(:);
  n_of = length(xx);

  pa = cell2mat(pp);
  [~,dup_idx] = intersect(tt(1:3,:)',pa(1:3,:)','rows');
  for ii = dup_idx(:)'
    seg_fn = sprintf(...
        'tmp_pb_psd_%d-%d-%d-%s.h5', ...
        tt(1:3,ii)', datestr(now,30));
    dvid_conn.get_segmentation(...
        tt(1:3,ii)' - vol_of, vol_sz, seg_fn, seg_name, ...
        false, true, false);
    ss = read_image_stack(seg_fn);
    delete(seg_fn);

    got_match = false;
    for kk=1:n_of
      if( ss( yy(kk),xx(kk),zz(kk) ) ~= ...
          ss(vol_of(2)+1,vol_of(1)+1,vol_of(3)+1) )
        continue
      end
      nx = tt(1,ii) + xx(kk) - vol_of(1);
      ny = tt(2,ii) + yy(kk) - vol_of(2);
      nz = tt(3,ii) + zz(kk) - vol_of(3);
      if( ismember([nx ny nz],pa(1:3,:)','rows') )
        continue
      end

      got_match = true;
      tt(1:3,ii) = [nx ny nz]';
      break
    end

    if(~got_match), keyboard, end

  end

end
