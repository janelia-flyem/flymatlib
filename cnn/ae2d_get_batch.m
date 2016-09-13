function [data, labels] = ae2d_get_batch(imdb, batch)

  ii = 1; % training
  if(min(batch) > imdb.nums(1))
    ii = 2; % validation
  end

  n_examples   = length(batch);

  patch_sz     = imdb.patch_sz;
  corrupt_type = imdb.corrupt_type;

  if(corrupt_type == 0)
    num_corrupt = floor(patch_sz^2 * imdb.corrupt);
  end
  if(corrupt_type == 1)
    sgm_corrupt = imdb.corrupt;
  end

  d_sz   = size(imdb.data{ii});
  xx     = ceil( (d_sz(1)-patch_sz+1)*rand(n_examples,1) );
  yy     = ceil( (d_sz(2)-patch_sz+1)*rand(n_examples,1) );
  zz     = ceil( (d_sz(3))           *rand(n_examples,1) );

  data   = zeros(patch_sz,patch_sz,1,...
                 n_examples, 'single');
  labels = zeros(patch_sz,patch_sz,1,...
                 n_examples, 'single');

  if(imdb.data_aug)
    aug_rot = floor(4*rand(n_examples,1));
    aug_ref = floor(2*rand(n_examples,1));
  else
    aug_rot = zeros(n_examples,1);
    aug_ref = zeros(n_examples,1);
  end

  for jj=1:n_examples
    data_tmp = imdb.data{ii}(...
        xx(jj):xx(jj)+patch_sz-1,...
        yy(jj):yy(jj)+patch_sz-1,...
        zz(jj));
    if(aug_rot(jj))
      for kk=1:size(data_tmp,3)
        data_tmp(:,:,kk) = rot90( data_tmp(:,:,kk),aug_rot(jj));
      end
    end
    if(aug_ref(jj))
      for kk=1:size(data_tmp,3)
        data_tmp(:,:,kk) = fliplr(data_tmp(:,:,kk));
      end
    end

    labels(:,:,1,jj) = data_tmp;
    if(corrupt_type == 0)
      rr = randsample(patch_sz^2, num_corrupt);
      data_tmp(rr) = 0;
    end
    if(corrupt_type == 1)
      data_tmp = data_tmp + ...
          sgm_corrupt * randn(patch_sz,patch_sz);
    end
    data(:,:,1,jj) = data_tmp;
  end

end
