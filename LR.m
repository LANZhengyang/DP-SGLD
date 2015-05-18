clear all; close all; clc;

set(0,'defaulttextinterpreter','latex');

s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);

%Setting for data simulation
N = 100; % data size
D = 3; % parameter size
betaTrue = randi([-1 1],D,1);

% % Add correlation to design matrix X
muDesg = zeros(D-1,1);
corrX = 0.7;
SigmaDesg = zeros(D-1,D-1);
for i = 1:D-1
    for j = i:D-1
        SigmaDesg(i,j) = corrX^(j-i);
        SigmaDesg(j,i) = SigmaDesg(i,j);
    end
end

% Prior
muStar = zeros(D,1);
SigmaStar = eye(D);
invSigmaStar = inv(SigmaStar);

% vary values of epsilon
nEps = 20;
logEps = linspace(-3,3,nEps);
epsilon = exp(logEps);

nRepeat = 100;
loglikSGLD = zeros(nRepeat,1);
loglikDPSGLD = zeros(nRepeat,nEps);

accSGLD = zeros(nRepeat,1);
accDPSGLD = zeros(nRepeat,nEps);

tic;

for l = 1:nRepeat
    %Simulate data
    %X = mvnrnd(muDesg,SigmaDesg,n);
    X = [ones(N,1),mvnrnd(muDesg,SigmaDesg,N)];
    probTrue = exp(X*betaTrue)./(1+exp(X*betaTrue));
    y = zeros(N,1);
    for j=1:N
        y(j)=binornd(1,probTrue(j));
    end
    
    % Run SGLD for logistic regression
    SGLDsamples = LR_SGLD(X,y);

    % Run DP-SGLD for logistic regression
    DPSGLDsamples = LR_DP_SGLD(X,y,epsilon);

    % Generate test data with the same data size N
    XTest = [ones(N,1),mvnrnd(muDesg,SigmaDesg,N)];
    probTrueTest = exp(XTest*betaTrue)./(1+exp(XTest*betaTrue));
    yTest = zeros(N,1);
    for j=1:N
        yTest(j)=binornd(1,probTrueTest(j));
    end
    
    % Generate beta
 
    % Prediction results for SGLD
    betaSamples = SGLDsamples;
    probPred = mean(exp(XTest*betaSamples')./(1+exp(XTest*betaSamples')),2);
    loglikSGLD(l) = sum(yTest.*log(probPred)+(1-yTest).*log(1-probPred));
    accSGLD(l) = mean((probPred > 0.5) == yTest);

    
     % Prediction results for DP-SGLD for different epsilon
    for k = 1:nEps
        betaSamples = DPSGLDsamples(:,:,k);
        probPred = mean(exp(XTest*betaSamples')./(1+exp(XTest*betaSamples')),2);
        loglikDPSGLD(l,k) = sum(yTest.*log(probPred)+(1-yTest).*log(1-probPred));
        accDPSGLD(l,k) = mean((probPred > 0.5) == yTest);
    end
end

time = toc/nRepeat

currenttime= datestr(now,'dd-mm-yy_HH_MM_SS');

% Loglikelihood: compare prediction results of SGLD and DP-SGLD 
loglikSGLDMean = repmat(mean(loglikSGLD),1,nEps);
loglikSGLDMeanStd = repmat(std(loglikSGLD)/sqrt(nRepeat),1,nEps);

loglikDPSGLDMean  = mean(loglikDPSGLD,1);
loglikDPSGLDMeanStd = std(loglikDPSGLD,1)/sqrt(nRepeat);

figure(1);
errorbar(logEps,loglikSGLDMean,loglikSGLDMeanStd);
hold on;
errorbar(logEps,loglikDPSGLDMean,loglikDPSGLDMeanStd);
title('Log-likelihood');
legend('SGLD','DP-SGLD');

% save figures
set(figure(1), 'paperpositionmode', 'auto');
filename= ['LR_loglik_' currenttime '.eps'];
print('-depsc',filename);

% Prediction accuracy: compare prediction results of SGLD and DP-SGLD 
accSGLDMean = repmat(mean(accSGLD),1,nEps);
accSGLDMeanStd = repmat(std(accSGLD)/sqrt(nRepeat),1,nEps);

accDPSGLDMean  = mean(accDPSGLD,1);
accDPSGLDMeanStd = std(accDPSGLD,1)/sqrt(nRepeat);

figure(2);
errorbar(logEps,accSGLDMean,accSGLDMeanStd);
hold on;
errorbar(logEps,accDPSGLDMean,accDPSGLDMeanStd);
title('Accuracy');
legend('SGLD','DP-SGLD');

% save figures
set(figure(2), 'paperpositionmode', 'auto');
filename= ['LR_accuracy_' currenttime '.eps'];
print('-depsc',filename);

