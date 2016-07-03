function tbar_psd_json_raveler2dvid(fn_in, fn_out, vol_tot_2)

  [tlocs,plocs] = tbar_json2locs(fn_in, [], true);

  tlocs(2,:) = vol_tot_2 - tlocs(2,:) - 1;
  for ii=1:length(plocs)
    if(~isempty(plocs{ii}))
      plocs{ii}(2,:) = vol_tot_2 - plocs{ii}(2,:) - 1;
    end
  end

  tbar_psd_json_write(fn_out, tlocs, plocs);
end
