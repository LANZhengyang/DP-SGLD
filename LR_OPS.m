function samples = LR_OPS(X,y,B,epsilon,maxIter,burnIn,stepSize)

if nargin < 5
    maxIter = 10000;
    burnIn = 0.5*maxIter;
    stepSize = 1;
end

plotInd = 0;

%Setting for data simulation
D = size(X,2);  % parameter size
rho = min(1,epsilon/4/B);

% prior
muStar = zeros(D,1);
SigmaStar = eye(D);
invSigmaStar = inv(SigmaStar);

%% Initialize
beta0 = rand(1,D);
%beta0 = zeros(1,D);
betaVec = [];
betaVec(1,:) = beta0;

% simple random walk Metropolis
for k = 1:maxIter
    curBeta = betaVec(k,:)';
    
    %propBeta = mvnrnd(zeros(D,1),diag(repmat(stepSize,1,D)),1)';
    propBeta = mvnrnd(curBeta,diag(repmat(stepSize,1,D)),1)';
    
    
    logProp = rho*(y'*(X*propBeta)-sum(log(1+exp(X*propBeta)))...
        -0.5*(propBeta-muStar)'*invSigmaStar*(propBeta-muStar));
    logCur = rho*(y'*(X*curBeta)-sum(log(1+exp(X*curBeta)))...
        -0.5*(curBeta-muStar)'*invSigmaStar*(curBeta-muStar));
    logRatio = logProp - logCur;
    
    a = min(1,exp(logRatio));
    
    if rand(1) < a
        betaVec(k+1,:) = propBeta';
    else
        betaVec(k+1,:) = curBeta';
    end
end
 
if plotInd == 1
    figure;
    plotDim = ceil(sqrt(D));
    for k = 1:D
        subplot(plotDim,plotDim,k);
        plot(betaVec(:,k));
        title(['OPS $\beta_' num2str(k) '$']);
    end
end
 
samples = betaVec(burnIn+1:end,:);

end
