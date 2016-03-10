function s = fml_get_exe(single_threaded)
% s = FML_GET_EXE(single_threaded)
% return string containing full path of script to run
%   compiled flymatlib executable
% 
% use single_threaded = true (false by default) to return
%   single threaded version

  fmlpath = fileparts(fileparts(mfilename('fullpath')));

  if(exist('single_threaded','var') && ...
     single_threaded > 0)
    s = [fmlpath '/fml_dist/my_run_fml_dist_single.sh'];
  else
    s = [fmlpath '/fml_dist/my_run_fml_dist.sh'];
  end

end
