function fml_make_mex(use_gpu, use_openmp)
% FML_MAKE_MEX(use_gpu, use_openmp) to compile third-party mex files
% set use_gpu to true to enable gpu support (default false)
% set use_openmp for mexconv3d (default false)

  if(~exist('use_gpu') || isempty(use_gpu))
    use_gpu = false;
  end
  if(~exist('use_openmp') || isempty(use_openmp))
    use_openmp = false;
  end

  % compile matconvnet
  if(use_gpu)
    vl_compilenn('enableGpu', true);
  else
    vl_compilenn();
  end

  make_all(use_gpu, use_openmp); % compile mexconv3d
end
