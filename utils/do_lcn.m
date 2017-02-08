function im_lcn = do_lcn(im, lcn_sz, lcn_sg, lcn_std)
  if(~exist('lcn_sz','var') || isempty(lcn_sz))
    lcn_sz = 201;
  end
  if(~exist('lcn_sg','var') || isempty(lcn_sg))
    lcn_sg = 100;
  end
  if(~exist('lcn_std','var') || isempty(lcn_std))
    lcn_std = 3.4;
  end

  lcn_fl  = fspecial('gaussian', lcn_sz, lcn_sg);
  lcn_sz_m1d2 = (lcn_sz-1)/2;

  im  = single(im);
  imf = padarray(im, [lcn_sz_m1d2, lcn_sz_m1d2], 'circular');

  im_mn  = conv2fft(imf, lcn_fl, 'valid');
  im_mn  = padarray(im_mn, [lcn_sz_m1d2, lcn_sz_m1d2], 'circular');

  im_sn  = imf - im_mn;
  im_std = sqrt(conv2fft(im_sn.^2, lcn_fl, 'valid'));

  im_stds = mean(im_std(:));
  im_lcn = im_sn(lcn_sz_m1d2+1:end-lcn_sz_m1d2,...
                 lcn_sz_m1d2+1:end-lcn_sz_m1d2) ./ ...
           max( im_std, im_stds );

  im_lcn = im_lcn * 128 / lcn_std + 127;
  im_lcn(im_lcn < 0)   = 0;
  im_lcn(im_lcn > 255) = 255;
  im_lcn = uint8(im_lcn);

end
