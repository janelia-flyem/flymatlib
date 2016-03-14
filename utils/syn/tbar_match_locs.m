function mm = tbar_match_locs(dists, allow_mult)
% TBAR_MATCH_LOCS match predict and groundtruth t-bar locations
% mm = TBAR_MATCH_LOCS(dists, allow_mult)
%
% allow_mult = 1
%   allows predicted locations to match multiple ground-truth locs
%   defaults to 0 if not specified
  
  if(~exist('allow_mult','var') || isempty(allow_mult))
    allow_mult = false;
  end
  
  dists_flipped = false;
  if(size(dists,2) > size(dists,1))
    dists_flipped = true;
    dists = dists';
  end
  
  num_predict     = size(dists,1);
  num_groundtruth = size(dists,2);
  num_vars        = num_predict*num_groundtruth;
  
  A = zeros(num_predict+num_groundtruth,num_vars);
  b = ones( num_predict+num_groundtruth,1);
  
  jj = 0;
  if(~allow_mult || dists_flipped)
    for ii=1:num_predict
      jj = jj+1;
      A(jj, ii:num_predict:num_vars) = 1;
    end
  end
  if(~allow_mult || ~dists_flipped)
    for ii=0:num_predict:num_vars-1
      jj = jj+1;
      A(jj, ii+(1:num_predict)) = 1;
    end
  end
  
  idx    = find(dists<=0);
  n_vars = length(idx);
  
  opts = optimoptions('intlinprog','Display','off');
  mm_sub = intlinprog(dists(idx), 1:n_vars, ...
                      A(:,idx), b, [], [], ...
                      zeros(n_vars,1), ones(n_vars,1), ...
                      opts);
  
  mm      = zeros(size(dists));
  mm(idx) = mm_sub;
  
  if(dists_flipped)
    mm = mm';
  end
end
