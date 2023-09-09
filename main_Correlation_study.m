% Code studies impact of correlation among sources on algorithms. 
% Code generates Fig. 5 (a) and (b) in R. R. Pote and B. D. Rao, "Maximum
% Likelihood-Based Gridless DoA Estimation Using Structured Covariance
% Matrix Recovery and SBL With Grid Refinement," in IEEE Transactions on
% Signal Processing, vol. 71, pp. 802-815, 2023.

% % Written by Rohan R. Pote, 2022

clear all; close all
addpath ./lib/
addpath ./lib/RAM_code/

%% Parameters
rng(1)
K = 2; % no. of sources
M = 6; % no. of sensors
L = 500; % no. of snapshots
SNR = 20; % in dB
P = [1 1];
mmITER = 20;
rho_abs=[0:0.1:0.9 0.99]; lrho = rand+1i*rand;
mgrid = 4096; % MUSIC grid size
ITER = 50; % random realizations for averaging
spos = 0:M-1;
Mapt=spos(end)+1; % Aperture size; also the max. total number of lags
So = eye(Mapt,Mapt); So = So(spos+1, :); % observed samples selection matrix
W = P(2)/(10^(SNR/10)); lambda = W;
% Source locations
source_u=[-1/4 1/4];
psi = pi*source_u;

%% Experiment

SCM_doaerrorSqrd_iter = zeros(ITER, length(rho_abs),K);
FbSCM_doaerrorSqrd_iter = zeros(ITER, length(rho_abs),K);
StructCovMLE_doaerrorSqrd_iter = zeros(ITER, length(rho_abs),K);
% RAM_doaerrorSqrd_iter = zeros(ITER, length(rho_abs),K);
GLSPRW_doaerrorSqrd_iter = zeros(ITER, length(rho_abs),K);
GLSPC_doaerrorSqrd_iter = zeros(ITER, length(rho_abs),K);
for iter = 1:ITER
    s = randn(K,L(end))+1j*randn(K,L(end));
    n = randn(Mapt,L(end))+1j*randn(Mapt,L(end));
    for rhoiter=1:length(rho_abs)
        rho = rho_abs(rhoiter)*lrho/abs(lrho);
        sigma = toeplitz(rho.^(0:K-1));
        [Vs, Ds] = eig(sigma);
        y = exp(1j*(0:Mapt-1)'*psi)*(Vs*Ds^0.5*Vs')*diag(sqrt(P/2))*s(:, 1:L)+sqrt(W/2)*n(:, 1:L); 
        yo = So*y;
        Ryoyo = yo*yo'/L;
        %% SCM
        [~,~,u_est]=SCM_MUSIC(Ryoyo,K,spos,mgrid);
        err_mat = repmat(u_est, 1, K)-repmat(source_u, K, 1);
        [~, ind_vec] = min(abs(err_mat));
        err_vec(1)=err_mat(ind_vec(1),1); err_vec(2)=err_mat(ind_vec(2),2);
        SCM_doaerrorSqrd_iter(iter, rhoiter,1) = err_vec(1);
        SCM_doaerrorSqrd_iter(iter, rhoiter,2) = err_vec(2);
        %% Forward-Backward Averaging
        J = fliplr(eye(M));
        S = (Ryoyo+J*conj(Ryoyo)*J)/2; % Forward-Backward averaging
        [~,~,u_est]=SCM_MUSIC(S,K,spos,mgrid);
        err_mat = repmat(u_est, 1, K)-repmat(source_u, K, 1);
        [~, ind_vec] = min(abs(err_mat));
        err_vec(1)=err_mat(ind_vec(1),1); err_vec(2)=err_mat(ind_vec(2),2);
        FbSCM_doaerrorSqrd_iter(iter, rhoiter,1) = err_vec(1);
        FbSCM_doaerrorSqrd_iter(iter, rhoiter,2) = err_vec(2);
        %% StructCovMLE-Proposed Algorithm
        [~,~,u_est]=StructCovMLE_MUSIC(Ryoyo,K,spos,mgrid,So,mmITER,lambda);
        err_mat = repmat(u_est, 1, K)-repmat(source_u, K, 1);
        [~, ind_vec] = min(abs(err_mat));
        err_vec(1)=err_mat(ind_vec(1),1); err_vec(2)=err_mat(ind_vec(2),2);
        StructCovMLE_doaerrorSqrd_iter(iter, rhoiter,1) = err_vec(1);
        StructCovMLE_doaerrorSqrd_iter(iter, rhoiter,2) = err_vec(2);
        %% RAM (Courtesy: Prof. Zai Yang)
%         [Q1, ~] = qr(yo',0);
%         yoecon = yo*Q1; Lecon = size(yoecon, 2);
%         n_eta = sqrt(W*(M*L+2*sqrt(M*L)));
%         [U_all, ~, freq, ~] = RAM_sdpt3(yoecon, spos+1, Mapt, n_eta, K);
%         U=U_all{end};
%         Rhat = (U(Lecon+1:end,Lecon+1:end)+U(Lecon+1:end,Lecon+1:end)')/2;
%         u_est = sort(freq{end}/pi);
%         err_mat = repmat(u_est, 1, K)-repmat(source_u, K, 1);
%         [~, ind_vec] = min(abs(err_mat));
%         err_vec(1)=err_mat(ind_vec(1),1); err_vec(2)=err_mat(ind_vec(2),2);
%         RAM_doaerrorSqrd_iter(iter, rhoiter,1) = err_vec(1);
%         RAM_doaerrorSqrd_iter(iter, rhoiter,2) = err_vec(2);
        %% Formulation in Eq. 30 (Paper: a compact formulation for the l_{2,1} mixed-norm minimization problem)
        lambda_glsprw = sqrt(W*M*log(M));
        [~,~,u_est]=GL_SPARROW_MUSIC(Ryoyo,K,spos,mgrid,So,lambda_glsprw);
        err_mat = repmat(u_est, 1, K)-repmat(source_u, K, 1);
        [~, ind_vec] = min(abs(err_mat));
        err_vec(1)=err_mat(ind_vec(1),1); err_vec(2)=err_mat(ind_vec(2),2);
        GLSPRW_doaerrorSqrd_iter(iter, rhoiter,1) = err_vec(1);
        GLSPRW_doaerrorSqrd_iter(iter, rhoiter,2) = err_vec(2);
        %% Gridless SPICE
        [~,~,u_est]=GL_SPICE_MUSIC(Ryoyo,K,spos,mgrid,So,L);
        err_mat = repmat(u_est, 1, K)-repmat(source_u, K, 1);
        [~, ind_vec] = min(abs(err_mat));
        err_vec(1)=err_mat(ind_vec(1),1); err_vec(2)=err_mat(ind_vec(2),2);
        GLSPC_doaerrorSqrd_iter(iter, rhoiter,1) = err_vec(1);
        GLSPC_doaerrorSqrd_iter(iter, rhoiter,2) = err_vec(2);
    end
end
% Source 1
figure
ax=plot(rho_abs, mean(SCM_doaerrorSqrd_iter(:,:,1),1),'--r', 'LineWidth',1); hold on;
plot(rho_abs, mean(FbSCM_doaerrorSqrd_iter(:,:,1),1),'-b','LineWidth',1)
plot(rho_abs, mean(StructCovMLE_doaerrorSqrd_iter(:,:,1),1),'-og','LineWidth',2,'MarkerSize',20)
% plot(rho_abs, mean(RAM_doaerrorSqrd_iter(:,:,1),1),':k','LineWidth',2)
plot(rho_abs, mean(GLSPRW_doaerrorSqrd_iter(:,:,1),1),'-s','Color',[0.93 0.69 0.13],'LineWidth',2,'MarkerSize',10)
plot(rho_abs, mean(GLSPC_doaerrorSqrd_iter(:,:,1),1),'-dm','LineWidth',2,'MarkerSize',10)
hold on
axis([0 0.99 -2*1e-3 2*1e-3])
grid on
xticks([0:0.1:0.9 0.99])
yticks([-2*1e-3:0.5*1e-3:2*1e-3])
xlabel('Absolute value of correlation coefficient |\rho|')
ylabel('Bias in u-space for source 1')
legend('SCM','FB SCM','StructCovMLE','GL SPARROW','GL SPICE', 'Location','southeast')
ax = ax.Parent;   % Important
set(ax,'FontWeight','bold','FontSize',14)
% Source 2
figure
ax=plot(rho_abs, mean(SCM_doaerrorSqrd_iter(:,:,2),1),'--r', 'LineWidth',1); hold on;
plot(rho_abs, mean(FbSCM_doaerrorSqrd_iter(:,:,2),1),'-b','LineWidth',1)
plot(rho_abs, mean(StructCovMLE_doaerrorSqrd_iter(:,:,2),1),'-og','LineWidth',2,'MarkerSize',20)
% plot(rho_abs, mean(RAM_doaerrorSqrd_iter(:,:,2),1),':k','LineWidth',2)
plot(rho_abs, mean(GLSPRW_doaerrorSqrd_iter(:,:,2),1),'-s','Color',[0.93 0.69 0.13],'LineWidth',2,'MarkerSize',10)
plot(rho_abs, mean(GLSPC_doaerrorSqrd_iter(:,:,2),1),'-dm','LineWidth',2,'MarkerSize',10)
hold on
axis([0 0.99 -2*1e-3 2*1e-3])
grid on
xticks([0:0.1:0.9 0.99])
yticks([-2*1e-3:0.5*1e-3:2*1e-3])
xlabel('Absolute value of correlation coefficient |\rho|')
ylabel('Bias in u-space for source 2')
legend('SCM','FB SCM','StructCovMLE','GL SPARROW','GL SPICE', 'Location','northeast')
ax = ax.Parent;   % Important
set(ax,'FontWeight','bold','FontSize',14)