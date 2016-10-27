function [ll_t, ll_p] = get_labels(this, seg_name, tt_pts, pp_pts)

  global DFEVAL_DIR

  if(exist('pp_pts','var') && ~isempty(pp_pts))
    get_psds = true;
  else
    get_psds = false;
  end

  while(true)
    tmp_out = sprintf('%s/tmp_pts_in_%s_%s.json', ...
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

  if(~get_psds)
    all_pts = tt_pts(1:3,:);
  else
    pp_pts_mat = cell2mat(pp_pts);
    if(~isempty(pp_pts_mat))
      all_pts = [ tt_pts(1:3,:) pp_pts_mat(1:3,:) ];
    else
      all_pts = tt_pts(1:3,:);
    end
  end

  n_pts = size(all_pts,2);
  fid   = fopen(tmp_out, 'w');
  fprintf(fid, '[');
  for ii=1:n_pts
    fprintf(fid, '[%d,%d,%d]',all_pts(1:3,ii));
    if(ii~=n_pts), fprintf(fid,','); end
  end
  fprintf(fid, ']\n');
  fclose(fid);

  dvid_cmd = ...
      sprintf(['curl -s -f -X GET ' ...
               '"%s/api/node/%s/%s/labels?%s" ' ...
               '-d "@%s" > %s'], ...
              this.machine_name, this.repo_name, ...
              seg_name, ...
              this.user_string, ...
              tmp_out, tmp_in);
  this.run_dvid_cmd(dvid_cmd);

  ss = fileread(tmp_in);
  dd = cell2mat(parse_json(ss));

  n_tt = size(tt_pts,2);
  ll_t = dd(1:n_tt);

  ll_p = {};
  if(get_psds)
    n_pp = length(pp_pts);
    ll_p = cell(1,n_pp);
    idx  = n_tt;
    for ii=1:n_pp
      n_pii    = size(pp_pts{ii},2);
      ll_p{ii} = dd(idx + (1:n_pii));
      idx      = idx + n_pii;
    end
  end

  system(sprintf('rm %s', tmp_out));
  system(sprintf('rm %s', tmp_in));
end
