function tbar_json_raveler2dvid(fn_in, fn_out, vol_tot_2)

  locs = tbar_json2locs(fn_in, [], true);
  
  locs(2,:) = vol_tot_2 - locs(2,:);
  
  tbar_json_write(fn_out, locs);
end
