function rr = in_roi(this, roi_name, pts)

  global DFEVAL_DIR

  while(true)
    tmp_out = sprintf('%s/tmp_pts_out_%s_%s.json', ...
                     DFEVAL_DIR, datestr(now,30), ...
                     get_random_id([],1));
    if(exist(tmp_out,'file')==0)
      system(sprintf('touch %s', tmp_out));
      break
    end
    pause(10);
  end
  while(true)
    tmp_in = sprintf('%s/tmp_pts_in_%s_%s.json', ...
                     DFEVAL_DIR, datestr(now,30), ...
                     get_random_id([],1));
    if(exist(tmp_in,'file')==0)
      system(sprintf('touch %s', tmp_in));
      break
    end
    pause(10);
  end

  n_pts = size(pts,2);
  fid   = fopen(tmp_out, 'w');
  fprintf(fid, '[');
  for ii=1:n_pts
    fprintf(fid, '[%d,%d,%d]',pts(1:3,ii));
    if(ii~=n_pts), fprintf(fid,','); end
  end
  fprintf(fid, ']\n');
  fclose(fid);

  dvid_cmd = ...
      sprintf(['curl -s -f -X POST ' ...
               '"%s/api/node/%s/%s/ptquery?%s" ' ...
               '-d "@%s" > %s'], ...
              this.machine_name, this.repo_name, ...
              roi_name, ...
              this.user_string, ...
              tmp_out, tmp_in);
  this.run_dvid_cmd(dvid_cmd);

  ss = fileread(tmp_in);
  rr = cell2mat(parse_json(ss));

  system(sprintf('rm %s', tmp_out));
  system(sprintf('rm %s', tmp_in));
end
