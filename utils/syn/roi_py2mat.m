function roi_py2mat(fn_in, fn_out)

  fid = fopen(fn_in);
  ss = fscanf(fid,'%d,%d,%d,%d',Inf);
  fclose(fid);
  ss = reshape(ss,4,[])';

  fid = fopen(fn_out, 'w');
  for jj=1:size(ss,1)
    fprintf(fid, '%d %d %d %d %d %d\n', ...
            ss(jj,4), ss(jj,3), ss(jj,2), ...
            ss(jj,1), ss(jj,1), ss(jj,1));
  end
  fclose(fid);

end
