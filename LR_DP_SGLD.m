function samples = LR_DP_SGLD(X,y,epsilon)

% plot index
plotInd = 0;

N = size(X,1); % data size
D = size(X,2); % parameter size

tau = floor(sqrt(N)); % size of minibatch
% epsilon = 0.5;
delta = 0.1;
L = 0.8; % Lipschitz constant
%T = epsilon^2*N/(32*tau*log(2/delta))+1;
T = 2*N;
alpha = 0.5;
maxIter = floor(N*T/tau);

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

% simple random walk Metropolis
for t = 1:maxIter
    
    % random sample a minibatch
    S = randsample(N,tau);
    
    % sample coordinates of z
    eta(t) = alpha*epsilon^2/(128*L^2*log(2.5*N*T/tau/delta)*log(2/delta)*t);
    etaBound = alpha*eta(t)/t*N*T/tau;
    zVar = max(eta(t),etaBound);
    
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

burnIn = maxIter - (1-alpha)*N*T/tau;
samples = betaVec(burnIn+1:end,:);

end
