function samples = LR_DP_SGLD(X,y,epsilon)

% plot index
plotInd = 0;

N = size(X,1); % data size
D = size(X,2); % parameter size

tau = floor(sqrt(N)); % size of minibatch
% epsilon = 0.5;
nEps = size(epsilon,2);
delta = 0.1;
L = 0.8; % Lipschitz constant
%T = epsilon^2*N/(32*tau*log(2/delta))+1;
T = 2*N;
alpha = 0.1;
%maxIter = floor(N*T/tau);
maxIter = 10000;

% Initialization with OPS
B = 0.8;
epsOPS = 0.1;
betaInitOPS = LR_OPS(X,y,B,epsOPS);
%betaInitOPS = LR_OPS(X,y,B,epsilon);

% prior
muStar = zeros(D,1);
SigmaStar = eye(D);
invSigmaStar = inv(SigmaStar);

%% Initialize
beta0 = betaInitOPS(end,:);
%beta0 = rand(1,D);
%beta0 = zeros(1,D);
betaArray = [];
betaArray(1,:,:) = permute(repmat(beta0,nEps,1),[3 2 1]);

eta = zeros(maxIter,1);
z = zeros(maxIter,D);

% simple random walk Metropolis
for t = 1:maxIter
    
    % random sample a minibatch
    S = randsample(N,tau);
    
    % sample coordinates of z
    % eta(t) = a*(b+t)^(-gamma);
    for i = 1:nEps
        
        eta(t) = alpha*epsilon(i)^2/(128*L^2*log(2.5*N*T/tau/delta)*log(2/delta)*t);
        etaBound = alpha*eta(t)/t*N*T/tau;
        zVar = max(eta(t),etaBound);
        
        for j = 1:D
            z(t,j) = normrnd(0,sqrt(zVar));
        end
        gradR = invSigmaStar*(betaArray(t,:,i)'-muStar);
        gradL = -X(S,:)'*(y(S)-exp(X(S,:)*betaArray(t,:,i)')./(1+exp(X(S,:)*betaArray(t,:,i)')));
        
        
        betaArray(t+1,:,i) = betaArray(t,:,i)-eta(t)*(gradR+N/tau*gradL)'+z(t,:);
    end
    
    a = 1;
end

if plotInd == 1
    for i = 1:nEps
        figure;
        plotDim = ceil(sqrt(D));
        for j = 1:D
            subplot(plotDim,plotDim,j);
            plot(betaArray(:,j,i));
            title(['$\beta_' num2str(j) '$ ($\epsilon = $' num2str(epsilon(i)) ')']);
        end
    end
end

burnIn = maxIter - (1-alpha)*N*T/tau;
samples = betaArray(burnIn+1:end,:,:);
end
