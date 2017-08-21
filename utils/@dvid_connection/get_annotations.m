function [tt,pp] = get_annotations(this, vol_start, vol_sz, ...
                                         annotations_name, ...
                                         is_groundtruth, bf, ...
                                         exclude_multi)

  global DFEVAL_DIR

  if(~exist('is_groundtruth','var') || isempty(is_groundtruth))
    is_groundtruth = false;
  end
  if(~exist('bf','var') || isempty(bf))
    bf = 70; % buffer to get psds attached to T-bars
  end
  if(~exist('exclude_multi','var') || isempty(exclude_multi))
    exclude_multi = false;
  end

  if(nargout > 1)
    return_psd = true;
  else
    return_psd = false;
    bf         = 0;
  end

  while(true)
    tmp_fn = sprintf('%s/tmp_annot_%s_%s.json', ...
                     DFEVAL_DIR, datestr(now,30), ...
                     get_random_id([],1));
    if(exist(tmp_fn,'file')==0)
      system(sprintf('touch %s', tmp_fn));
      break
    end
    pause(10);
  end

  use_roi = false;
  if(isempty(vol_sz))
    use_roi  = true;
    roi_name = vol_start;
  end

  if(~use_roi)
    vol_start_bf = vol_start - bf;
    vol_sz_bf    = vol_sz + 2*bf;
  end

  if(use_roi)
    dvid_cmd = ...
        sprintf(['%s GET ' ...
                 '"%s/api/node/%s/%s/roi/' ...
                 '%s?%s" > %s'], ...
                this.http_cmd, ...
                this.machine_name, this.repo_name, ...
                annotations_name, ...
                roi_name, ...
                this.user_string, tmp_fn);
  else
    dvid_cmd = ...
        sprintf(['%s GET ' ...
                 '"%s/api/node/%s/%s/elements/' ...
                 '%d_%d_%d/%d_%d_%d?%s" > %s'], ...
                this.http_cmd, ...
                this.machine_name, this.repo_name, ...
                annotations_name, ...
                vol_sz_bf(1),    vol_sz_bf(2),    vol_sz_bf(3),    ...
                vol_start_bf(1), vol_start_bf(2), vol_start_bf(3), ...
                this.user_string, tmp_fn);
  end
  this.run_dvid_cmd(dvid_cmd);

  ss    = fileread(tmp_fn);
  if(isequal(ss,'null'))
    tt = zeros(4,0);
    pp = {};
    return
  end

  dd    = parse_json(ss);
  n_dd  = length(dd);
  system(sprintf('rm %s', tmp_fn));

  tt     = zeros(4, n_dd);
  n_psd  = zeros(1, n_dd);
  idx    = 0;
  tt_idx = containers.Map();

  for ii=1:n_dd
    if(~strcmp(dd{ii}.Kind,'PreSyn')), continue, end
    if(exclude_multi && isfield(dd{ii}.Prop, 'annotation') && ...
       strcmp(dd{ii}.Prop.annotation, 'Multi'))
      continue
    end
    if(is_groundtruth && ...
       (isempty(dd{ii}.Prop.user) || dd{ii}.Prop.user(1)=='$'))
      warning('FML:Warning', ...
              'ignoring automated prediction T-bar');
      continue
    end
    tt_pos      = cell2mat(dd{ii}.Pos);

    if(~use_roi)
      if(tt_pos(1) <  vol_start(1) || ...
         tt_pos(1) >= vol_start(1) + vol_sz(1) || ...
         tt_pos(2) <  vol_start(2) || ...
         tt_pos(2) >= vol_start(2) + vol_sz(2) || ...
         tt_pos(3) <  vol_start(3) || ...
         tt_pos(3) >= vol_start(3) + vol_sz(3))
        continue
      end
    end

    idx = idx + 1;
    tt(1:3,idx) = tt_pos;
    if(isfield(dd{ii}.Prop,'conf'))
      tt(4,  idx) = str2double(dd{ii}.Prop.conf);
    else
      warning('FML:Warning', 'no confidence value, setting to 1');
      tt(4,  idx) = 1;
    end
    n_psd( idx) = length(dd{ii}.Rels);

    if(return_psd)
      tt_idx(sprintf('%d_', tt(1:3,idx))) = idx;
    end
  end
  n_tt = idx;
  tt   = tt(:,1:n_tt);

  if(~return_psd), return, end

  pp = cell(1,n_tt);

  for ii=1:n_dd
    if(~strcmp(dd{ii}.Kind,'PostSyn')), continue, end
    if(is_groundtruth && ...
       (isempty(dd{ii}.Prop.user) || dd{ii}.Prop.user(1)=='$'))
      warning('FML:Warning', ...
              'ignoring automated prediction PSD');
      continue
    end
    if(isempty(dd{ii}.Rels))
      warning('FML:Warning', 'disconnected PSD');
      continue
    end
    assert(length(dd{ii}.Rels)==1 && ...
           isequal(dd{ii}.Rels{1}.Rel, 'PostSynTo'), ...
           'FML:AssertionFailed', ...
           'unexpected PostSyn Rel');

    tt_pos = cell2mat(dd{ii}.Rels{1}.To);
    tt_str = sprintf('%d_', tt_pos);
    if(~tt_idx.isKey(tt_str)), continue, end

    idx    = tt_idx(tt_str);
    jj     = size(pp{idx},2) + 1;

    pp{idx}(1:3,jj) = cell2mat(dd{ii}.Pos);
    if(isfield(dd{ii}.Prop, 'conf'))
      pp{idx}(4,  jj) = str2double(dd{ii}.Prop.conf);
    else
      warning('FML:Warning', 'no confidence value, setting to 1');
      pp{idx}(4,  jj) = 1;
    end
  end

  for ii=1:n_tt
    if(n_psd(ii)~=size(pp{ii},2))
      warning('FML:Warning', ...
              'expected num PSDs mismatch: %d, %d', ...
              n_psd(ii), size(pp{ii},2));
    end
  end
end
