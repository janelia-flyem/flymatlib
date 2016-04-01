function fml_setup()
% FML_SETUP()
%   add flymatlib to the matlab path

  global DFEVAL_DIR
  if(isempty(DFEVAL_DIR) || ...
     ~exist(DFEVAL_DIR,'dir'))
    warning('FML:Warning', ...
            'DFEVAL_DIR invalid, set to: "%s"', DFEVAL_DIR);
  end

  fmlpath = fileparts(mfilename('fullpath'));

  % add third party functions
  run(fullfile(fmlpath, 'third_party', ...
               'matconvnet', 'matlab', 'vl_setupnn.m'));
  run(fullfile(fmlpath, 'third_party', ...
               'mexconv3d', 'setup_path.m'));

  addpath(fmlpath);
  addpath(fullfile(fmlpath, 'cnn'                       ));
  addpath(fullfile(fmlpath, 'cnn',         'core'       ));
  addpath(fullfile(fmlpath, 'utils'                     ));
  addpath(fullfile(fmlpath, 'utils',       'syn'        ));
  addpath(fullfile(fmlpath, 'third_party'               ));
  addpath(fullfile(fmlpath, 'third_party', 'export_fig' ));
  addpath(fullfile(fmlpath, 'examples'                  ));
end
