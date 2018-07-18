function seg = get_segmentation(...
    this, vol_start, vol_sz, seg_fn, seg_name, ...
    use_compression, do_permute, use_throttle)

  global DFEVAL_DIR
  while(true)
    tmp_fn = sprintf('%s/tmp_segm_%d-%d-%d_%s_%s.json', ...
                     DFEVAL_DIR, vol_start, ...
                     datestr(now,30), get_random_id([],1));
    if(exist(tmp_fn,'file')==0)
      system(sprintf('touch %s', tmp_fn));
      break
    end
    pause(10);
  end

  if(~exist('use_compression','var'))
    use_compression = true; % default to use lz4
  end
  if(~exist('do_permute','var'))
    do_permute = true; % to matlab ordering
  end
  if(~exist('use_throttle','var'))
    use_throttle = true; % to matlab ordering
  end

  if(use_compression)
    compress_str = '&compression=lz4';
  else
    compress_str = '';
  end
  throttle_str = '';
  if(use_throttle)
    throttle_str = '&throttle=on';
  end
  dvid_cmd = ...
      sprintf(['%s GET ' ...
               '"%s/api/node/%s/%s/raw/0_1_2/' ...
               '%d_%d_%d/%d_%d_%d?supervoxels=true&%s%s%s" > %s'], ...
              this.http_cmd, ...
              this.machine_name, this.repo_name, seg_name, ...
              vol_sz(1),    vol_sz(2),    vol_sz(3),    ...
              vol_start(1), vol_start(2), vol_start(3), ...
              this.user_string, throttle_str, compress_str, ...
              tmp_fn);
  this.run_dvid_cmd(dvid_cmd);

  fid = fopen(tmp_fn);
  assert(fid>0, 'FLYEMLIB:AssertionFailed', ...
         sprintf('could not read file: %s', tmp_fn));

  if(use_compression)
    seg = fread(fid, Inf, 'uint8=>uint8');
    seg = fml_lz4_mex(prod(vol_sz)*8, seg);
    seg = typecast(seg, 'uint64');
  else
    seg = fread(fid, Inf, 'uint64=>uint64');
  end

  fclose(fid);
  delete(tmp_fn);

  % reshape and switch from DVID to Matlab indexing order
  seg = reshape(seg, vol_sz);
  if(do_permute)
    seg = permute(seg, [2 1 3]);
  end

  chunk_sz = min([50 50 50], vol_sz);

  if(~isempty(seg_fn))
    if(exist(seg_fn, 'file')), delete(seg_fn); end
    h5create(seg_fn, '/main', size(seg), ...
             'Datatype', 'uint64', ...
             'ChunkSize', chunk_sz, ...
             'Deflate', 4, ...
             'Shuffle', 1);
    h5write(seg_fn, '/main', seg);
  end
end
