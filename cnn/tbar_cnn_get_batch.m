function [data, labels] = tbar_cnn_get_batch(imdb, batch)
% [data, labels] = TBAR_CNN_GET_BATCH(imdb, batch)
% called within tbar_cnn_train
% returns data and labels for one mini-batch,
%   generated on the fly

n_substacks = length(imdb.data);
iiss = 1:(n_substacks-1); % training
if(imdb.use_all_data)
  iiss = 1:n_substacks;
end
if(min(batch) > imdb.nums(1))
  iiss = n_substacks; % validation
end

patch_sz  = imdb.patch_sz;
ext_lower = imdb.ext_lower;
ext_upper = imdb.ext_upper;

corrupt_type = imdb.corrupt_type;

if(corrupt_type == 0)
  num_corrupt = floor(patch_sz^3 * imdb.corrupt);
end
if(corrupt_type == 1)
  sgm_corrupt = imdb.corrupt;
end

n_label_out = size(imdb.labels{1},4);

n_examples_total = length(batch);
data   = zeros(patch_sz,patch_sz,patch_sz,1,...
               n_examples_total, 'single');
labels = ones(1,1,1, n_label_out,...
              n_examples_total, 'single');
idx = 0;

ratios_sum(iiss) = sum(imdb.ratios(iiss,:),2);
n_examples(iiss) = round(...
    n_examples_total * ratios_sum(iiss) );
n_examples(iiss(end)) = n_examples_total - ...
    sum(n_examples(iiss(1:end-1)));
% n_examples = n_examples_total/length(iiss);
% assert(n_examples == round(n_examples), ...
%        'FLYEMLIB:AssertionFailed',...
%        ['number of training substacks should ' ...
%         'evenly divide batch size']);

n_ratios      = length(imdb.classes);

for ii=iiss
  num_tot  = 0;
  num_per_class = zeros(1,n_ratios);
  for cc = 1:n_ratios-1
    num_per_class(cc) = round(...
        n_examples(ii) * imdb.ratios(ii,cc)/ratios_sum(ii));
    num_tot = num_tot + num_per_class(cc);
  end
  num_per_class(n_ratios) = n_examples(ii) - num_tot;

  for cc = 1:n_ratios
    if(num_per_class(cc)==0), continue, end
    pts = randsample(imdb.pts{ii, cc}, num_per_class(cc), true);
    %length(pts)<num_per_class(cc+1));

    [xa,ya,za] = ind2sub(size(imdb.data{ii}), pts);

    n_pts = length(pts);
    for jj = 1:n_pts
      idx = idx + 1;
      xx = xa(jj); yy = ya(jj); zz = za(jj);

      data_tmp = imdb.data{ii}(...
          xx-ext_lower:xx+ext_upper,...
          yy-ext_lower:yy+ext_upper,...
          zz-ext_lower:zz+ext_upper);
      if(corrupt_type == 0)
        rr = randsample(patch_sz^3, num_corrupt);
        data_tmp(rr) = 0;
      end
      if(corrupt_type == 1)
        data_tmp = data_tmp + ...
            sgm_corrupt * randn(patch_sz,patch_sz,patch_sz);
      end
      data(:,:,:,1,idx) = data_tmp;
      if(imdb.is_autoencoder)
        labels(1,1,1,1,idx) = imdb.data{ii}(xx,yy,zz);
      else
        labels(1,1,1,:,idx) = imdb.labels{ii}(xx,yy,zz,:);
      end
    end
  end
end

if(imdb.data_aug)
  nn = size(data,5);
  aug_rot = floor(4*rand(nn,1));
  aug_ref = floor(2*rand(nn,1));
  aug_fpz = floor(2*rand(nn,1));

  for ii=1:nn
    if(aug_rot(ii))
      for zz=1:size(data,3)
        data(:,:,zz,1,ii) = rot90( data( :,:,zz,1,ii),aug_rot(ii));
      end
    end
    if(aug_ref(ii))
      data(:,:,:,1,ii) = fliplr(data(:,:,:,1,ii));
    end
    if(aug_fpz(ii))
      data(:,:,:,1,ii) = flip(data(:,:,:,1,ii),3);
    end
  end

  if(size(labels,4)>1) % also need to update x,y,z offsets
    for ii=1:nn
      if(aug_rot(ii))
        switch aug_rot(ii)
          case 1
            labels(1,1,1,2:3,ii) = ...
                [-labels(1,1,1,3,ii), ...
                  labels(1,1,1,2,ii)];
          case 2
            labels(1,1,1,2:3,ii) = -labels(1,1,1,2:3,ii);
          case 3
            labels(1,1,1,2:3,ii) = ...
                [ labels(1,1,1,3,ii), ...
                 -labels(1,1,1,2,ii)];
        end
      end
      if(aug_ref(ii))
        labels(1,1,1,3,ii) = -labels(1,1,1,3,ii);
      end
      if(aug_fpz(ii))
        labels(1,1,1,4,ii) = -labels(1,1,1,4,ii);
      end
    end
  end

  if(size(labels,4)>1) % also need to update x,y,z offsets
    for ii=1:nn
      if(aug_rot(ii))
        switch aug_rot(ii)
          case 1
            labels(1,1,1,2:3,ii) = ...
                [-labels(1,1,1,3,ii), ...
                  labels(1,1,1,2,ii)];
          case 2
            labels(1,1,1,2:3,ii) = -labels(1,1,1,2:3,ii);
          case 3
            labels(1,1,1,2:3,ii) = ...
                [ labels(1,1,1,3,ii), ...
                 -labels(1,1,1,2,ii)];
        end
      end
      if(aug_ref(ii))
        labels(1,1,1,3,ii) = -labels(1,1,1,3,ii);
      end
    end
  end
end
