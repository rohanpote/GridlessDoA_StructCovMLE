% Code studies impact of more sources than sensors on algorithms. 
% Code generates Fig. 4 in R. R. Pote and B. D. Rao, "Maximum
% Likelihood-Based Gridless DoA Estimation Using Structured Covariance
% Matrix Recovery and SBL With Grid Refinement," in IEEE Transactions on
% Signal Processing, vol. 71, pp. 802-815, 2023.

% % Written by Rohan R. Pote, 2022

clear all; close all
addpath ./lib/
addpath ./lib/RAM_code/

%% Parameters
rng(1)
K = 8; % no. of sources
M = 6; % no. of sensors
L = 4;
SNR = 20; % in dB
P = 1; % source power
mmITER = 20;
rho_abs=0; lrho = rand+1i*rand; rho = rho_abs*lrho/abs(lrho);
mgrid = 4096; % MUSIC evaluate grid size
ITER = 20; % random realizations for averaging
% Nested array sensor positions
if mod(M,2)==0
    M1=M/2; M2=M/2;
else
    M1=(M-1)/2; M2=(M+1)/2;
end
spos = union(0:M1-1,M1+(0:M2-1)*(M1+1));
Mapt=spos(end)+1; % Aperture size; also the max. total number of lags
So = eye(Mapt,Mapt); So = So(spos+1, :); % observed samples selection matrix
W = P/(10^(SNR/10)); lambda = W; 
sigma = toeplitz(rho.^(0:K-1));
[Vs, Ds] = eig(sigma);
% Source locations
source_u=-1+1/K:2/K:1-1/K; % in u-space
psi = pi*source_u;
%% Experiment

SCM_doaerrorSqrd_iter = zeros(ITER, length(L));
StructCovMLE_doaerrorSqrd_iter = zeros(ITER, length(L));
% RAM_doaerrorSqrd_iter = zeros(ITER, length(L));
GLSPRW_doaerrorSqrd_iter = zeros(ITER, length(L));
GLSPC_doaerrorSqrd_iter = zeros(ITER, length(L));
for iter = 1:ITER
    s = randn(K,L(end))+1j*randn(K,L(end));
    n = randn(Mapt,L(end))+1j*randn(Mapt,L(end));
    for liter=1:length(L)
        y = exp(1j*(0:Mapt-1)'*psi)*(Vs*Ds^0.5*Vs')*diag(sqrt(P/2))*s(:, 1:L(liter))+sqrt(W/2)*n(:, 1:L(liter)); 
        yo = So*y;
        Ryoyo = yo*yo'/L(liter);
        %% SCM
        [u_grid,Smusic,u_est]=SCM_MUSIC(Ryoyo,K,spos,mgrid);
        figure(1)
        plot(u_grid, 10*log10(Smusic/max(Smusic)),'LineWidth', 2); hold on; grid on
        err_mat = repmat(u_est, 1, K)-repmat(source_u, K, 1);
        [err_vec, ind_vec] = min(abs(err_mat));
        SCM_doaerrorSqrd_iter(iter, liter) = mean(err_vec.^2);
        %% StructCovMLE-Proposed Algorithm
        [u_grid,Smusic,u_est]=StructCovMLE_MUSIC(Ryoyo,K,spos,mgrid,So,mmITER,lambda);
        figure(2)
        plot(u_grid, 10*log10(Smusic/max(Smusic)),'LineWidth', 2); hold on; grid on
        err_mat = repmat(u_est, 1, K)-repmat(source_u, K, 1);
        [err_vec, ind_vec] = min(abs(err_mat));
        StructCovMLE_doaerrorSqrd_iter(iter, liter) = mean(err_vec.^2);
        %% RAM (Courtesy: Prof. Zai Yang)
%         [Q1, ~] = qr(yo',0);
%         yoecon = yo*Q1; Lecon = size(yoecon, 2);
%         n_eta = sqrt(lambda*(M*L(liter)+2*sqrt(M*L(liter))));
%         [U_all, ~, freq, ~] = RAM_sdpt3(yoecon, spos+1, Mapt, n_eta, K);
%         U=U_all{end};
%         Rhat = (U(Lecon+1:end,Lecon+1:end)+U(Lecon+1:end,Lecon+1:end)')/2;
%         [u_grid,Smusic,~]=root_MUSIC(Rhat,K,0:Mapt-1,mgrid);
%         figure(3)
%         plot(u_grid, 10*log10(Smusic/max(Smusic)),'LineWidth', 2); hold on; grid on
%         u_est = sort(freq{end}/pi);
%         err_mat = repmat(u_est, 1, K)-repmat(source_u, K, 1);
%         [err_vec, ind_vec] = min(abs(err_mat));
%         RAM_doaerrorSqrd_iter(iter, liter) = mean(err_vec.^2);
        %% Formulation in Eq. 30 (Paper: a compact formulation for the l_{2,1} mixed-norm minimization problem)
        lambda_glsprw = sqrt(lambda*M*log(M)); %
        [u_grid,Smusic,u_est]=GL_SPARROW_MUSIC(Ryoyo,K,spos,mgrid,So,lambda_glsprw);
        figure(4)
        plot(u_grid, 10*log10(Smusic/max(Smusic)),'LineWidth', 2); hold on; grid on
        err_mat = repmat(u_est, 1, K)-repmat(source_u, K, 1);
        [err_vec, ind_vec] = min(abs(err_mat));
        GLSPRW_doaerrorSqrd_iter(iter, liter) = mean(err_vec.^2);
        %% Gridless SPICE
        [u_grid,Smusic,u_est]=GL_SPICE_MUSIC(Ryoyo,K,spos,mgrid,So,L(liter));
        figure(5)
        plot(u_grid, 10*log10(Smusic/max(Smusic)),'LineWidth', 2); hold on; grid on
        err_mat = repmat(u_est, 1, K)-repmat(source_u, K, 1);
        [err_vec, ind_vec] = min(abs(err_mat));
        GLSPC_doaerrorSqrd_iter(iter, liter) = mean(err_vec.^2);
    end
end

figure(1)
plot(repmat(source_u,100,1),repmat(linspace(-100,0,100)',1,K),'--r','LineWidth',1)
axis([-1 1 -100 0])
xlabel('Spatial angle in u-space')
ylabel('(normalized) Pseeudospectrum in dB')
title(['RMSE=' num2str(sqrt(mean(SCM_doaerrorSqrd_iter)))])
set(gca, 'FontWeight', 'Bold')

figure(2)
plot(repmat(source_u,100,1),repmat(linspace(-100,0,100)',1,K),'--r','LineWidth',1)
axis([-1 1 -100 0])
xlabel('Spatial angle in u-space')
ylabel('(normalized) Pseeudospectrum in dB')
title(['RMSE=' num2str(sqrt(mean(StructCovMLE_doaerrorSqrd_iter)))])
set(gca, 'FontWeight', 'Bold')

% figure(3)
% plot(repmat(source_u,100,1),repmat(linspace(-100,0,100)',1,K),'--r','LineWidth',1)
% axis([-1 1 -100 0])
% xlabel('Spatial angle in u-space')
% ylabel('(normalized) Pseeudospectrum in dB')
% title(['RMSE=' num2str(sqrt(mean(RAM_doaerrorSqrd_iter)))])
% set(gca, 'FontWeight', 'Bold')

figure(4)
plot(repmat(source_u,100,1),repmat(linspace(-100,0,100)',1,K),'--r','LineWidth',1)
axis([-1 1 -100 0])
xlabel('Spatial angle in u-space')
ylabel('(normalized) Pseeudospectrum in dB')
title(['RMSE=' num2str(sqrt(mean(GLSPRW_doaerrorSqrd_iter)))])
set(gca, 'FontWeight', 'Bold')

figure(5)
plot(repmat(source_u,100,1),repmat(linspace(-100,0,100)',1,K),'--r','LineWidth',1)
axis([-1 1 -100 0])
xlabel('Spatial angle in u-space')
ylabel('(normalized) Pseeudospectrum in dB')
title(['RMSE=' num2str(sqrt(mean(GLSPC_doaerrorSqrd_iter)))])
set(gca, 'FontWeight', 'Bold')