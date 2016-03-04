X = randn(5,6,1,3);
c = 2*(randn(5,6,1,3)>0)-1;
c(2,3,1,1) = 0;
c(5,1,1,2) = 0;
c(4,4,1,3) = 0;

dd  = 1e-5;
tol = 1e-5;

tt = 0.9;

obj = fml_nnsigentloss(X,c,tt);
dx  = fml_nnsigentloss(X,c,tt,1);

for ii=1:5
  for jj=1:6
    for kk=1:3
      X2 = X;
      X2(ii,jj,1,kk) = X(ii,jj,1,kk)+dd;
      obj2 = fml_nnsigentloss(X2,c,tt);
      dx_n = (obj2-obj)/dd;
      fprintf('[%g]', abs(dx(ii,jj,1,kk)-dx_n));
      assert( abs(dx(ii,jj,1,kk)-dx_n) < tol );
    end
  end
end
fprintf('\n');
