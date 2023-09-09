% Code studies impact of low number of snapshots on algorithms.
% Code generates Fig. 3 in R. R. Pote and B. D. Rao, "Maximum
% Likelihood-Based Gridless DoA Estimation Using Structured Covariance
% Matrix Recovery and SBL With Grid Refinement," in IEEE Transactions on
% Signal Processing, vol. 71, pp. 802-815, 2023.

% % Written by Rohan R. Pote, 2022

clear all; close all
addpath ./lib/
%% Parameters
rng(1)
% Parameters
K = 2; % no. of sources
M = 10; % no. of sensors
L = 1; % single snapshot
SNR = 20; % in dB
P = [1 1]; % source power
mmITER = 20; %% Majorization-minimization iteration count
rho_abs=0; lrho = rand+1i*rand; rho = rho_abs*lrho/abs(lrho); % uncorrelated sources
mgrid = 4096; % MUSIC evaluate grid size
ITER = 10; % random realizations for averaging
spos = 0:M-1; % sensor positions (ULA)
Mapt=spos(end)+1; % Aperture size; also the max. total number of lags
So = eye(Mapt,Mapt); So = So(spos+1, :); % observed samples selection matrix
W = P(2)/(10^(SNR/10)); lambda = W; % noise variance assumed known; can be estimated in general
sigma = toeplitz(rho.^(0:K-1));
[Vs, Ds] = eig(sigma);
% Source locations
source_u=[-1/Mapt 1/Mapt]; % in u-space
psi = pi*source_u;

%% Experiment
for iter = 1:ITER
    s = randn(K,L)+1j*randn(K,L);
    n = randn(Mapt,L)+1j*randn(Mapt,L);
    y = exp(1j*(0:Mapt-1)'*psi)*(Vs*Ds^0.5*Vs')*diag(sqrt(P/2))*s(:, 1:L)+sqrt(W/2)*n(:, 1:L);
    yo = So*y; % observed measurements- for ULA, all measurements are observed
    Ryoyo = yo*yo'/L;
    %% SCM
    [u_grid,Smusic,u_est]=SCM_MUSIC(Ryoyo,K,spos,mgrid);
    figure(1)
    plot(u_grid, 10*log10(Smusic/max(Smusic)),'LineWidth', 2); hold on; grid on
    %% Forward-Backward Averaging
    [u_grid,Smusic,u_est]=Fb_SCM_MUSIC(Ryoyo,K,spos,mgrid);
    figure(2)
    plot(u_grid, 10*log10(Smusic/max(Smusic)),'LineWidth', 2); hold on; grid on
    %% StructCovMLE-Proposed Algorithm
    [u_grid,Smusic,u_est]=StructCovMLE_MUSIC(Ryoyo,K,spos,mgrid,So,mmITER,lambda);
    figure(3)
    plot(u_grid, 10*log10(Smusic/max(Smusic)),'LineWidth', 2); hold on; grid on
end
figure(1)
plot(repmat(source_u,100,1),repmat(linspace(-90,0,100)',1,K),'--r','LineWidth',1)
axis([-1 1 -90 0])
xlabel('Spatial angle in u-space')
ylabel('(normalized) Pseeudospectrum in dB')
set(gca, 'FontWeight', 'Bold')

figure(2)
plot(repmat(source_u,100,1),repmat(linspace(-90,0,100)',1,K),'--r','LineWidth',1)
axis([-1 1 -90 0])
xlabel('Spatial angle in u-space')
ylabel('(normalized) Pseeudospectrum in dB')
set(gca, 'FontWeight', 'Bold')

figure(3)
plot(repmat(source_u,100,1),repmat(linspace(-90,0,100)',1,K),'--r','LineWidth',1)
axis([-1 1 -90 0])
xlabel('Spatial angle in u-space')
ylabel('(normalized) Pseeudospectrum in dB')
set(gca, 'FontWeight', 'Bold')
