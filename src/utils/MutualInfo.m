function [Ixy,lambda]=MutualInfo(X,Y)
%%
% Estimating Mutual Information with Moon et al. 1995
% between X and Y
% Input parameter
% X and Y : data column vectors (nL*1, nL is the record length)
%
% Output 
% Ixy : Mutual Information
% lambda: scaled mutual information similar comparabble to
% cross-correlation coefficient
%
%  Programmed by 
%  Taesam Lee, Ph.D., Research Associate
%  INRS-ETE, Quebecc
%  Hydrologist 
%  Oct. 2010
%
%

X=X';
Y=Y';

d=2;
nx=length(X);
hx=(4/(d+2))^(1/(d+4))*nx^(-1/(d+4));

Xall=[X;Y];
sum1=0;
for is=1:nx

    pxy=p_mkde([X(is),Y(is)]',Xall,hx);
    px=p_mkde([X(is)],X,hx);
    py=p_mkde([Y(is)],Y,hx);
    sum1=sum1+log(pxy/(px*py));
end

Ixy=sum1/nx;

lambda=sqrt(1-exp(-2*Ixy));




%% Multivariate kernel density estimate using a normal kernel
% with the same h
% input data X : dim * number of records
%            x : the data point in order to estimate mkde (d*1) vector
%            h : smoothing parameter
function [pxy]=p_mkde(x,X,h);

s1=size(X);
d=s1(1);
N=s1(2);

Sxy=cov(X');
sum=0;
%p1=1/sqrt((2*pi)^d*det(Sxy))*1/(N*h^d);

invS=inv(Sxy);
detS=det(Sxy);
for ix=1:N
    p2=(x-X(:,ix))'*invS*(x-X(:,ix));
    sum=sum+1/sqrt((2*pi)^d*detS)*exp(-p2/(2*h^2));
end
pxy=1/(N*h^d)*sum;



%% Reference
%     Moon, Y. I., B. Rajagopalan, and U. Lall (1995), 
%     Estimation of Mutual Information Using Kernel Density Estimators, 
%     Phys Rev E, 52(3), 2318-2321.