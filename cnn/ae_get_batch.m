function [data, labels] = ae_get_batch(imdb, batch)

  n_substacks = length(imdb.data);
  iiss = 1:(n_substacks-1); % training
  if(min(batch) > imdb.nums(1))
    iiss = n_substacks; % validation
  end

  n_examples_total = length(batch);
  n_examples = n_examples_total/length(iiss);
  assert(n_examples == round(n_examples), ...
         'FLYEMLIB:AssertionFailed',...
         ['number of training substacks should ' ...
          'evenly divide batch size']);

  is3d         = imdb.is3d;
  patch_sz     = imdb.patch_sz;
  if(isscalar(patch_sz))
    patch_sz = patch_sz * ones(1,3);
  end
  if(~is3d), patch_sz(3) = 1; end
  label_sz     = imdb.label_sz;
  if(isscalar(label_sz))
    label_sz = label_sz * ones(1,3);
  end
  if(~is3d), label_sz(3) = 1; end
  corrupt_type = imdb.corrupt_type;

  if(corrupt_type == 0)
    num_corrupt = floor(prod(patch_sz) * imdb.corrupt);
  end
  if(corrupt_type == 1)
    sgm_corrupt = imdb.corrupt;
  end

  data   = zeros(patch_sz(1),patch_sz(2),patch_sz(3),1,...
                 n_examples_total, 'single');
  if(isempty(label_sz))
    labels = zeros(patch_sz(1),patch_sz(2),patch_sz(3),1,...
                   n_examples_total, 'single');
  else
    labels = zeros(label_sz(1),label_sz(2),label_sz(3),1,...
                   n_examples_total, 'single');
  end

  if(imdb.data_aug)
    aug_rot = floor(4*rand(n_examples_total,1));
    aug_ref = floor(2*rand(n_examples_total,1));
    aug_fpz = floor(2*rand(n_examples_total,1));
  else
    aug_rot = zeros(n_examples_total,1);
    aug_ref = zeros(n_examples_total,1);
    aug_fpz = zeros(n_examples_total,1);
  end

  jj = 0;
  for ii=iiss
    d_sz   = size(imdb.data{ii});
    xx     = ceil( (d_sz(1)-patch_sz(1)+1)*rand(n_examples,1) );
    yy     = ceil( (d_sz(2)-patch_sz(2)+1)*rand(n_examples,1) );
    zz     = ceil( (d_sz(3)-patch_sz(3)+1)*rand(n_examples,1) );

    for idx=1:n_examples
      jj = jj + 1;
      data_tmp = imdb.data{ii}(...
          xx(idx):xx(idx)+patch_sz(1)-1,...
          yy(idx):yy(idx)+patch_sz(2)-1,...
          zz(idx):zz(idx)+patch_sz(3)-1);
      if(aug_rot(jj))
        for kk=1:size(data_tmp,3)
          data_tmp(:,:,kk) = rot90( data_tmp(:,:,kk),aug_rot(jj));
        end
      end
      if(aug_ref(jj))
        data_tmp(:,:,:) = fliplr(data_tmp(:,:,:));
      end
      if(aug_fpz(jj))
        data_tmp(:,:,:) = flip(data_tmp(:,:,:),3);
      end

      if(isempty(label_sz))
        labels(:,:,:,1,jj) = data_tmp;
      else
        labels(:,:,:,1,jj) = reshape(...
            data_tmp(imdb.zero_mask), ...
            label_sz(1),label_sz(2),label_sz(3));
      end

      if(corrupt_type == 0)
        rr = randsample(prod(patch_sz), num_corrupt);
        data_tmp(rr) = 0;
      end
      if(corrupt_type == 1)
        data_tmp = data_tmp + ...
            sgm_corrupt * randn(patch_sz(1),patch_sz(2),patch_sz(3));
      end

      if(~isempty(imdb.zero_mask))
        data_tmp(imdb.zero_mask) = 0;
      end
      data(:,:,:,1,jj) = data_tmp;
    end
  end

  if(~is3d)
    data   = reshape(data, [patch_sz(1),patch_sz(2),1,...
                            n_examples_total]);
    lsz    = size(labels);
    labels = reshape(labels, [lsz(1),lsz(2),1,n_examples_total]);
  end
end
