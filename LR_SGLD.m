function samples = LR_SGLD(X,y)

maxIter = 10000;

% plot index
plotInd = 0;

N = size(X,1); % data size
D = size(X,2); % parameter size

tau = floor(sqrt(N)); % size of minibatch

% %step size
% a = 1;
% b = 1;
% gamma = 0.7;

% prior
muStar = zeros(D,1);
SigmaStar = eye(D);
invSigmaStar = inv(SigmaStar);

%% Initialize
beta0 = rand(1,D);
betaVec = [];
betaVec(1,:) = beta0;

eta = zeros(maxIter,1);
z = zeros(maxIter,D);

eta0 = 0.1;

% simple random walk Metropolis
for t = 1:maxIter
    
    % random sample a minibatch
    S = randsample(N,tau);
    
    % sample coordinates of z
    % eta(t) = a*(b+t)^(-gamma);
    eta(t) = max(1/(t+1),eta0);
    zVar = eta(t);
    
    for i = 1:D
        z(t,i) = normrnd(0,sqrt(zVar));
    end
    gradR = invSigmaStar*(betaVec(t,:)'-muStar);
    gradL = -X(S,:)'*(y(S)-exp(X(S,:)*betaVec(t,:)')./(1+exp(X(S,:)*betaVec(t,:)')));
    
    betaVec(t+1,:) = betaVec(t,:)-eta(t)*(gradR+N/tau*gradL)'+z(t,:);
    
end

if plotInd == 1
    figure;
    plotDim = ceil(sqrt(D));
    for k = 1:D
        subplot(plotDim,plotDim,k);
        plot(betaVec(:,k));
    end
end

burnIn = 0.5*maxIter;
samples = betaVec(burnIn+1:end,:);

end
