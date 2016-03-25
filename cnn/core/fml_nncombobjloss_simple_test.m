dd  = 1e-5;
tol = 1e-5;

X = randn(3,3,3,4,2);
c = zeros(size(X));
c(1,2,3,1,1) = 1;
c(1,2,3,3,1) = 0.5;
X(1,2,3,3,1) = -1;
X(1,2,3,[2 4],1) = 0;
c(2,3,1,1,2) = 1;
c(2,3,1,4,2) = -0.2;
X(2,3,1,2:4,2) = 0;

Y  = fml_nncombobjloss(X,c,1,0.9);
Y1 = fml_nnsigentloss(X(:,:,:,1,:), c(:,:,:,1,:), 1);
Y2 = (1.5/0.9 - 0.5 + 0.5*(0.2/0.9)^2)/27;

assert(abs(Y - (Y1+Y2)) < tol );

X = randn(5,6,3,4,3);
c = randn(5,6,3,4,3);
c(:,:,:,1,:) = 2*(c(:,:,:,1,:)>0)-1;
c(2,3,2,1,1) = 0;
c(5,1,3,1,2) = 0;
c(4,4,1,1,3) = 0;

tt = 0.9;
rnorm = 1.1;

obj = fml_nncombobjloss(X,c,tt,rnorm);
dx  = fml_nncombobjloss(X,c,tt,rnorm,1);

for ii=1:5
  for jj=1:6
    for kk=1:3
      for ll=1:4
        for mm=1:3
          X2 = X;
          X2(ii,jj,kk,ll,mm) = X(ii,jj,kk,ll,mm)+dd;
          obj2 = fml_nncombobjloss(X2,c,tt,rnorm);
          dx_n = (obj2-obj)/dd;
          fprintf('[%g]', abs(dx(ii,jj,kk,ll,mm)-dx_n));
          assert( abs(dx(ii,jj,kk,ll,mm)-dx_n) < tol );
        end
      end
    end
  end
end
fprintf('\n');
