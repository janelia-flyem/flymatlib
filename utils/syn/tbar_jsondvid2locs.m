function locs = tbar_jsondvid2locs(fn)
  tt = jsondecode(fileread(fn));

  idx = 0;
  locs = zeros(4, size(tt,1));

  for ii=1:size(tt,1)
    if( ~isequal(tt(ii).Kind, 'PreSyn') ), continue, end

    idx = idx + 1;
    locs(1:3,idx) = tt(ii).Pos;
    cc = tt(ii).Prop.conf;
    if(ischar(cc))
      cc = str2double(cc);
    end
    locs(4,idx) = cc;
  end

  locs = locs(:,1:idx);
end
