X = randn(5,6,1,3);
c = randn(5,6,1,3);

dd  = 1e-5;
tol = 1e-5;

l_eps = 0.1;

obj = fml_nnl1loss(X,c,l_eps,2);
dx  = fml_nnl1loss(X,c,l_eps,2,1);

for ii=1:5
  for jj=1:6
    for kk=1:3
      X2 = X;
      X2(ii,jj,1,kk) = X(ii,jj,1,kk)+dd;
      obj2 = fml_nnl1loss(X2,c,l_eps,2);
      dx_n = (obj2-obj)/dd;
      fprintf('[%g]', abs(dx(ii,jj,1,kk)-dx_n));
      assert( abs(dx(ii,jj,1,kk)-dx_n) < tol );
    end
  end
end
fprintf('\n');
