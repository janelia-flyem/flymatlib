function [locs, psd_locs] = tbar_json2locs(fn, offset, get_conf, ...
                                           get_body)
% TBAR_JSON2LOCS read json file of tbar locations into matrix
% [locs, psd_locs] = TBAR_JSON2LOCS(...
%                      fn, offset, get_conf, get_body)
%
% fn         json filename
% offset     shift locations
% get_conf   return confidence as fourth row
% get_body   return body ID as fifth row

  if(~exist('get_conf','var') || isempty(get_conf))
    get_conf = false;
  end
  if(~exist('get_body','var') || isempty(get_body))
    get_body = false;
  end

  ss = fileread(fn);
  dd = parse_json(ss);

  nn = length(dd.data);

  if(get_body)
    locs = zeros(5, nn);
  else
    if(get_conf)
      locs = zeros(4, nn);
    else
      locs = zeros(3, nn);
    end
  end

  for ii=1:nn
    locs(1:3,ii) = cell2mat(dd.data{ii}.T_bar.location);
    if(get_conf)
      cc = dd.data{ii}.T_bar.confidence;
      if(ischar(cc))
        cc = str2double(cc);
      end
      locs(4,  ii) = cc;
    end
    if(get_body)
      locs(5,  ii) = dd.data{ii}.T_bar.body_ID;
    end
  end

  if(exist('offset','var') && ~isempty(offset))
    if(isscalar(offset))
      locs(3,:)   = locs(3,:) - offset;
    else
      locs(1:3,:) = bsxfun(@minus, locs(1:3,:), offset');
    end
  end

  if(nargout == 2) % also output PSDs
    psd_locs = cell(1,nn);

    for ii=1:nn
      if(isfield(dd.data{ii},'partners'))
        mm = length(dd.data{ii}.partners);
      else
        mm = 0;
      end
      if(get_body)
        psd_locs{ii} = zeros(5, mm);
      else
        if(get_conf)
          psd_locs{ii} = zeros(4, mm);
        else
          psd_locs{ii} = zeros(3, mm);
        end
      end

      for jj=1:mm
        psd_locs{ii}(1:3,jj) = ...
            cell2mat(dd.data{ii}.partners{jj}.location);
        if(get_conf)
          psd_locs{ii}(4,jj) = ...
              dd.data{ii}.partners{jj}.confidence;
        end
        if(get_body)
          psd_locs{ii}(5,jj) = ...
              dd.data{ii}.partners{jj}.body_ID;
        end
      end
    end

    if(exist('offset','var') && ~isempty(offset))
      if(isscalar(offset))
        for ii=1:nn
          psd_locs{ii}(3,:) = psd_locs{ii}(3,:) - offset;
        end
      else
        for ii=1:nn
          psd_locs{ii}(1:3,:) = bsxfun(...
            @minus, psd_locs{ii}(1:3,:), offset');
        end
      end
    end

  end % end get PSDs
end
