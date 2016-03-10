function fn_out = fml_dfeval_worker(fn_in)
% fn_out = FML_DFEVAL_WORKER(fn_in)
% main entry point function for distributed computation with
%   compiled flymatlib Matlab executable

  fn_out = [fn_in(1:end-4), '_return.mat'];
  if(exist(fn_out, 'file'))
    % already ran successfully
    return
  end

  bundle = load(fn_in, 'bundle');
  bundle = bundle.bundle;
  
  f_out = cell(1,nargout(bundle.f));
  
  if(~isempty(f_out))
    [f_out{:}] = bundle.f(bundle.args{:}); %#ok<NASGU>

    n_whos = whos('f_out');
    if(n_whos.bytes>(2e9))
      save(fn_out, 'f_out', '-v7.3');
    else
      save(fn_out, 'f_out');
    end
  else
    bundle.f(bundle.args{:});
    system(sprintf('touch %s', fn_out));
  end
  
end
