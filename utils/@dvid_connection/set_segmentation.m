function set_segmentation(this, vol_offset, ...
                                seg_fn, seg_name, ...
                                chunk_sz, do_permute)

  if(ischar(seg_fn))
    seg = read_image_stack(seg_fn);
  else
    seg = seg_fn;
  end

  if(~exist('chunk_sz','var') || isempty(chunk_sz))
    chunk_sz = 256; % max to send over at once
  end
  if(~exist('do_permute','var') || isempty(do_permute))
    do_permute = true;
  end

  if(do_permute)
    seg = permute(seg, [2 1 3]); % put horizontal before vertical
  end

  if(~exist('vol_offset','var') || isempty(vol_offset))
    vol_offset = [0 0 0];
  end

  vol_sz     = size(seg);
  vol_sz_pad = 32*ceil(vol_sz / 32);

  seg_pad    = zeros(vol_sz_pad, 'uint64');
  seg_pad(1:vol_sz(1),1:vol_sz(2),1:vol_sz(3)) = seg;

  for ii=0:chunk_sz:vol_sz_pad(1)-1
    for jj=0:chunk_sz:vol_sz_pad(2)-1
      for kk=0:chunk_sz:vol_sz_pad(3)-1
        bin_fn = sprintf('%s_%s.bin', seg_name, datestr(now,30));
        fid    = fopen(bin_fn,'w');
        up_sz  = min( [chunk_sz, chunk_sz, chunk_sz], ...
                      vol_sz_pad - [ii jj kk] );
        fwrite(fid, seg_pad( ii+1:ii+up_sz(1), ...
                             jj+1:jj+up_sz(2), ...
                             kk+1:kk+up_sz(3) ), 'uint64');
        fclose(fid);

        dvid_cmd = sprintf(...
            ['curl -s -f -X POST --data-binary @%s ' ...
             '%s/api/node/%s/%s/raw/0_1_2/%d_%d_%d/%d_%d_%d'], ...
            bin_fn, this.machine_name, this.repo_name, seg_name, ...
            up_sz(1),         up_sz(2),         up_sz(3), ...
            vol_offset(1)+ii, vol_offset(2)+jj, vol_offset(3)+kk);
        st = system(dvid_cmd);

        if(st ~= 0)
          error('error connecting to dvid: %s', dvid_cmd);
        end

        delete(bin_fn);
      end
    end
  end

end
