function fml_make(single_threaded)
% FML_MAKE(single_threaded) to compile flymatlib

  if(exist('single_threaded','var') && ...
     single_threaded)
    mcc_out = '-o fml_dist_single -R -singleCompThread';
  else
    mcc_out = '-o fml_dist';
  end

  fmlpath = fileparts(mfilename('fullpath'));
  fprintf('changing directory to %s\n', fmlpath);
  wd = cd(fmlpath);

  mcc_cmd = sprintf(...
    ['mcc -m -v %s -R -nodesktop -d fml_dist/ ' ...
     'utils/fml_dfeval_worker.m ' ...
     '-a cnn -a utils ' ...
     '-a third_party/matconvnet ' ...
     '-a third_party/mexconv3d/mex_* ' ...
     '-a third_party/mexconv3d/util'], mcc_out);
  fprintf('%s\n', mcc_cmd);
  eval(mcc_cmd);

  cd(wd);
end
