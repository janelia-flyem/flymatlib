function set_image_segmentation(this, vol_offset, set_type, ...
                                      data_fn, data_name, ...
                                      chunk_sz, do_permute)

  block_sz = 32;

  switch(set_type)
    case 'image'
      data_type = 'uint8';
    case 'segmentation'
      data_type = 'uint64';
    otherwise
      assert(false, 'FML:AssertionFailed', ...
             'unknown set_type');
  end

  if(ischar(data_fn))
    data = read_image_stack(data_fn);
  else
    data = data_fn;
  end

  if(~exist('chunk_sz','var') || isempty(chunk_sz))
    chunk_sz = 256; % max to send over at once
  end
  if(~exist('do_permute','var') || isempty(do_permute))
    do_permute = true;
  end

  if(do_permute)
    data = permute(data, [2 1 3]); % put horizontal before vertical
  end

  if(~exist('vol_offset','var') || isempty(vol_offset))
    vol_offset = [0 0 0];
  end

  vol_offset_blk = block_sz*floor(vol_offset / block_sz);
  vol_offset_pad = vol_offset - vol_offset_blk;
  vol_sz         = size(data);
  vol_sz_off     = vol_sz + vol_offset_pad;
  vol_sz_pad     = block_sz*ceil(vol_sz_off / block_sz);

  data_pad       = zeros(vol_sz_pad, data_type);
  data_pad(vol_offset_pad(1)+(1:vol_sz(1)),...
           vol_offset_pad(2)+(1:vol_sz(2)),...
           vol_offset_pad(3)+(1:vol_sz(3))) = data;

  for ii=0:chunk_sz:vol_sz_pad(1)-1
    for jj=0:chunk_sz:vol_sz_pad(2)-1
      for kk=0:chunk_sz:vol_sz_pad(3)-1
        bin_fn = sprintf('%s_%s.bin', data_name, datestr(now,30));
        fid    = fopen(bin_fn,'w');
        up_sz  = min( [chunk_sz, chunk_sz, chunk_sz], ...
                      vol_sz_pad - [ii jj kk] );
        fwrite(fid, data_pad( ii+1:ii+up_sz(1), ...
                              jj+1:jj+up_sz(2), ...
                              kk+1:kk+up_sz(3) ), data_type);
        fclose(fid);

        dvid_cmd = sprintf(...
            ['curl -s -f -X POST --data-binary @%s ' ...
             '%s/api/node/%s/%s/raw/0_1_2/%d_%d_%d/%d_%d_%d'], ...
            bin_fn, this.machine_name, this.repo_name, data_name, ...
            up_sz(1), up_sz(2), up_sz(3), ...
            vol_offset_blk(1)+ii, ...
            vol_offset_blk(2)+jj, ...
            vol_offset_blk(3)+kk);
        st = system(dvid_cmd);

        if(st ~= 0)
          error('error connecting to dvid: %s', dvid_cmd);
        end

        delete(bin_fn);
      end
    end
  end

end
