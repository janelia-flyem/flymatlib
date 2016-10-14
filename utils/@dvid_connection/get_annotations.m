function [tt,dd] = get_annotations(this, vol_start, vol_sz, ...
                                         annotations_name)

  global DFEVAL_DIR

  while(true)
    tmp_fn = sprintf('%s/tmp_annot_%s_%s.mat', ...
                     DFEVAL_DIR, datestr(now,30), ...
                     get_random_id([],1));
    if(exist(tmp_fn,'file')==0)
      system(sprintf('touch %s', tmp_fn));
      break
    end
    pause(10);
  end

  dvid_cmd = ...
      sprintf(['%s GET ' ...
               '"%s/api/node/%s/%s/elements/' ...
               '%d_%d_%d/%d_%d_%d?%s" > %s'], ...
              this.http_cmd, ...
              this.machine_name, this.repo_name, ...
              annotations_name, ...
              vol_sz(1),    vol_sz(2),    vol_sz(3), ...
              vol_start(1), vol_start(2), vol_start(3), ...
              this.user_string, tmp_fn);
  this.run_dvid_cmd(dvid_cmd);

  ss    = fileread(tmp_fn);
  if(isequal(ss,'null'))
    tt = zeros(4,0);
    dd = {};
    return
  end

  dd    = parse_json(ss);
  n_dd  = length(dd);
  system(sprintf('rm %s', tmp_fn));

  tt    = zeros(4, n_dd);
  n_psd = zeros(1, n_dd);
  idx   = 0;
  for ii=1:n_dd
    if(~strcmp(dd{ii}.Kind,'PreSyn')), continue, end
    idx = idx + 1;
    tt(1:3,idx) = cell2mat(dd{ii}.Pos);
    tt(4,  idx) = str2double(dd{ii}.Prop.conf);
    n_psd       = length(dd{ii}.Rels);
  end
  tt = tt(:,1:idx);
end
