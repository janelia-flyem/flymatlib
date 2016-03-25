function [flt,dd] = fml_set_filter(radius)
  flt_sz = 2*radius+1;

  [xx, yy, zz] = ndgrid(...
    -radius:radius,-radius:radius,-radius:radius);
  dd = reshape(sqrt(xx(:).^2 + yy(:).^2 + zz(:).^2), ...
               [flt_sz flt_sz flt_sz]);
  
  flt = double(dd <= radius);
end
