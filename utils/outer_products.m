function op = outer_products(x, y)
% x, y : m x n
% op   : m x m x n

  [mx,n] = size(x);
  [my,n] = size(y);
  j     = 1:my;
  op    = reshape(repmat(x,my,1) .* ...
                  y(j(ones(mx,1),:),:), [mx my n]);
end
