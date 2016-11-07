function tbar_dvid2training(dvid_conn, ...
                            annotation, ...
                            vol_start, vol_sz, ...
                            get_image, im_norm, ...
                            radius_use, ...
                            dir_name, fn_prefix)

  if(get_image)
    % pull image, normalize
    fn_image = sprintf('%s/%simage.h5', dir_name, fn_prefix{1});
    dvid_conn.get_image(vol_start, vol_sz, fn_image, im_norm);
  end

  % get t-bar annotations
  tt = dvid_conn.get_annotations(...
      vol_start, vol_sz, annotation, ...
      true, 0, false);
  % convert to local coordinates (0-based)
  tt(1:3,:) = bsxfun(@minus, tt(1:3,:), vol_start');
  % write out ground-truth json
  fn_json = sprintf('%s/%ssynapses.json', dir_name, fn_prefix{3});
  tbar_json_write(fn_json, tt);
  % swap x-y
  tt(1:3,:) = tt([2 1 3],:);

  fn_out_lm = sprintf('%s/%s', dir_name, fn_prefix{2});
  % write out labels, mask by calling tbar_json2labelmask
  tbar_json2labelmask(tt, fn_out_lm, vol_sz, radius_use, ...
                      [],[],[],[],[],[]);
end
