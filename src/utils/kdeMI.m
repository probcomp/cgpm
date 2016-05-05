function I = kdeMI( X, Y )
%KDEMI computes the kernel density estimation
%   1. Input
%      X, Y: nx1-vectors
%
%   2. Output
%      I: mutual information estimation
%
% Author: Paolo Inglese <paolo.ingls@gmail.com>
%
% Reference: Moon, Young-Il, Balaji Rajagopalan, and Upmanu Lall.
%            "Estimation of mutual information using kernel density estimators."
%            Physical Review E 52.3 (1995): 2318-2321.

Xall = [X, Y];

numSamples = size(Xall, 1);
Itmp = zeros(numSamples, 1);
for i = 1:numSamples
    
    pXY = kde([X(i), Y(i)], Xall);
    pX = kde(X(i), X);
    pY = kde(Y(i), Y);
    
    Itmp(i) = pXY * log2(pXY / (pX * pY));
    
end

I = sum(Itmp);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function P = kde( x, y )
% kernel density estimation

numSamples = size(y, 1);
numVars = size(y, 2);

h = (4 / (numVars + 2))^(1/(numVars+4))*numSamples^(-1/(numVars+4));

y = bsxfun(@minus, y, mean(y));
S = y'*y / numSamples;
invS = pinv(S);

K = zeros(numSamples, 1);
for j = 1:numSamples
    u = (x - y(j, :))*invS*(x - y(j, :))' / h^2;
    K(j) = exp(-u/2) / ((2*pi)^(numVars/2)*h^numVars*det(S)^(1/2));
end
P = sum(K) / numSamples;

end




