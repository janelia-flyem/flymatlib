function fml_setup()
% FML_SETUP()
%   add flymatlib to the matlab path

  fmlp = fileparts(mfilename('fullpath'));
  
  addpath(fullfile(fmlp, 'cnn'        ));
  addpath(fullfile(fmlp, 'cnn', 'core'));
end
