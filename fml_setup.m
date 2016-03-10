function fml_setup()
% FML_SETUP()
%   add flymatlib to the matlab path

  fmlpath = fileparts(mfilename('fullpath'));
  
  % add third party functions
  run(fullfile(fmlpath, 'third_party', ...
               'matconvnet', 'matlab', 'vl_setupnn.m'));
  run(fullfile(fmlpath, 'third_party', ...
               'mexconv3d', 'setup_path.m'));
  
  addpath(fullfile(fmlpath, 'cnn'        ));
  addpath(fullfile(fmlpath, 'cnn', 'core'));
  addpath(fullfile(fmlpath, 'utils'      ));
end
