function r = get_random_id(max_val, as_str)
% GET_RANDOM_ID get random id (as string if desired)
% r = GET_RANDOM_ID(max_val, as_str)
  
  if(~exist('max_val','var') || isempty(max_val))
    max_val = 1e9;
  end
  if(~exist('as_str','var') || isempty(as_str))
    as_str = 0;
  end
  
  % s = rng('shuffle');
  r = ceil(max_val*rand());
  % rng(s);

  if(as_str)
    zero_pad = ceil(log10(max_val));
    cmd = sprintf('r = sprintf(''%%0%dd'', r);', zero_pad);
    eval(cmd);
  end
end
