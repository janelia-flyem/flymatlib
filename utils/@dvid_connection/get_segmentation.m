function seg = get_segmentation(...
    this, vol_start, vol_sz, seg_fn, seg_name)

  if(exist(seg_fn, 'file')), delete(seg_fn); end

  dvid_cmd = ...
      sprintf(['%s get ' ...
               '"%s/api/node/%s/%s/raw/0_1_2/' ...
               '%d_%d_%d/%d_%d_%d?throttle=on&%s" > %s'], ...
              this.http_cmd, ...
              this.machine_name, this.repo_name, seg_name, ...
              vol_sz(1),    vol_sz(2),    vol_sz(3),    ...
              vol_start(1), vol_start(2), vol_start(3), ...
              this.user_string, seg_fn);
  this.run_dvid_cmd(dvid_cmd);

  fid = fopen(seg_fn);
  assert(fid>0, 'FLYEMLIB:AssertionFailed', ...
         sprintf('could not read file: %s', seg_fn));

  seg = fread(fid, Inf, 'uint64=>uint64');

  fclose(fid);
  delete(seg_fn);

  % reshape and switch from DVID to Matlab indexing order
  seg = reshape(seg, vol_sz);
  seg = permute(seg, [2 1 3]);

  chunk_sz = min([50 50 50], vol_sz);

  h5create(seg_fn, '/main', size(seg), ...
           'Datatype', 'uint64', ...
           'ChunkSize', chunk_sz, ...
           'Deflate', 4, ...
           'Shuffle', 1);
  h5write(seg_fn, '/main', seg);

end
