% Code studies performance of SBL with Likelihood-based Grid Refinement
% when sources are off-grid
% Code generates Fig. 6 (a) and (b) in R. R. Pote and B. D. Rao, "Maximum
% Likelihood-Based Gridless DoA Estimation Using Structured Covariance
% Matrix Recovery and SBL With Grid Refinement," in IEEE Transactions on
% Signal Processing, vol. 71, pp. 802-815, 2023.

% % Written by Rohan R. Pote, 2022

clear all; close all
addpath ./lib/
addpath ./lib/MSBL_code/

%% Parameters
rng(1)
K = 2; % no. of sources
M = 6; % no. of sensors
L = 500; % no. of snapshots
SNR = 20; % in dB
P = 1; % source power
W = P/(10^(SNR/10)); % noise power
rho_abs=0; % absolute correlation coefficient
lrho = rand+1i*rand; rho = rho_abs*lrho/abs(lrho);
ITER = 50; % random realizations for averaging
% Sensor positions
spos = [0 1 2.1 3.5 4.7 10];
% Grid Parameters
sblgrid=150;
u_grid=-1+1/sblgrid:2/sblgrid:1-1/sblgrid;
Agrid=exp(1j*pi*spos'*u_grid);
% Sources' correlation (uncorrelated if rho_abs=0)
sigma = toeplitz(rho.^(0:K-1));
[Vs, Ds] = eig(sigma);

% Source locations
source_u=(-1+1/K:2/K:1)+(2*rand([1,K])-1)/K/10;
psi = pi*source_u;

%% Generating data, run MSBL, run grid refinement step
SBLbef_doaerrorSqrd_iter = zeros(ITER, 6); % before peak adjustment
SBLaft_doaerrorSqrd_iter = zeros(ITER, 6); % after peak adjustment
GridSize=zeros(ITER,6);

%% Stochastic CRB equal power sources assumed
Sigma_s=diag(sqrt(P))*sigma*diag(sqrt(P))';
num_factor=W/(2*L);
Acrb = exp(1j*pi*spos'*source_u);
Dcrb = 1j*spos'.*Acrb;
Ry_inv = eye(M)/((Acrb*Sigma_s*Acrb')+W*eye(M));
crb_psi = num_factor*eye(K)/(real(Dcrb'*(eye(M)-((Acrb/(Acrb'*Acrb))*Acrb'))*Dcrb).*((Sigma_s*Acrb'*Ry_inv*Acrb*Sigma_s).'));
sqrcrb_u_theta=real(sqrt(mean(diag((1/pi)*crb_psi*((1/pi).')))));

%% Proposed Algorithm
% figure(11); clf
for iter = 1:ITER
    iter
    s = randn(K,L(end))+1j*randn(K,L(end));
    n = randn(M,L(end))+1j*randn(M,L(end));
    tic
    %% Data Generation
    yo = sqrt(P/2)*exp(1j*spos'*psi)*(Vs*Ds^0.5*Vs')*s(:, 1:L)+sqrt(W/2)*n(:, 1:L);
    Ryoyo = yo*yo'/L;
    %% Run MSBL
    lambda = W;          % Initial value for the regularization parameter.
    learn_Lambda = 0;       % Not using its lambda learning rule to learn an (sub-)optimal lambda.
    GridSize(iter,1)=size(Agrid,2);
    [Weight,gamma_ind,gamma_est,count] = MSBL(Agrid,yo, lambda, learn_Lambda,'max_iters', 5000);
    [pks,locs]=findpeaks1(gamma_est);
    [mpks,mlocs]=maxk(pks, K);
    m_candidates=sort(locs(mlocs));% one source
    u_est=u_grid(m_candidates);
    SBLbef_doaerrorSqrd_iter(iter,1)=mean(abs(u_est-source_u).^2);
    %% Perform grid refinement
    % Parameters
    PRUNE_GAMMA=1e-3;
    g=3;
    resSBL=2/sblgrid;
    u_grid_updated=u_grid;
    Agrid_updated=Agrid;
    for itermultiresSBL=1:5
        [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,K,u_grid_updated,Agrid_updated,spos,M,Ryoyo,lambda);
        [pks,locs]=findpeaks1(gamma_est);
        [mpks,mlocs]=maxk(pks, K);
        K_est=length(mlocs);
        m_peaks=sort(locs(mlocs));% one source
        u_peaks=u_grid_updated(m_peaks);
        SBLaft_doaerrorSqrd_iter(iter,itermultiresSBL)=mean(abs(u_peaks-source_u).^2);
        u_grid_updated=u_grid_updated(gamma_est>PRUNE_GAMMA);
        resSBL=resSBL/g;
        u_new_grid=(-2*resSBL*g:resSBL:2*resSBL*g);
        u_new_grid=repmat(u_peaks,length(u_new_grid),1)+repmat(u_new_grid',1,K_est);
        
        % Discarding points in main grid to ensure uniform sampling
        % around peaks
        for iterPks=1:K_est
            u_grid_updated=u_grid_updated(u_grid_updated<u_new_grid(1,iterPks)|u_grid_updated>u_new_grid(end,iterPks));
        end
        
        u_new_grid=u_new_grid(:)';
        u_grid_updated=union(u_grid_updated,u_new_grid);
        
        % Removing redundant points feature (to be added)
        u_grid_updated=u_grid_updated(diff([-1 u_grid_updated])>=0.9*resSBL);
        
        Agrid_updated=exp(1j*pi*spos'*u_grid_updated);
        GridSize(iter,itermultiresSBL+1)=size(Agrid_updated,2);
        [Weight,gamma_ind,gamma_est,count] = MSBL(Agrid_updated,yo, lambda, learn_Lambda,'max_iters', 5000);
        
        [pks,locs]=findpeaks1(gamma_est);
        [mpks,mlocs]=maxk(pks, K);
        K_est=length(mlocs);
        m_peaks=sort(locs(mlocs));% one source
        u_peaks=u_grid_updated(m_peaks);
        SBLbef_doaerrorSqrd_iter(iter,itermultiresSBL+1)=mean(abs(u_peaks-source_u).^2);
    end
    [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,K,u_grid_updated,Agrid_updated,spos,M,Ryoyo,lambda);
    [pks,locs] = findpeaks1(gamma_est);
    [~, mlocs] = maxk(pks, K);
    ufinal_est = sort(u_grid_updated(locs(mlocs)))';
    %%
    err_mat = repmat(ufinal_est, 1, K)-repmat(source_u, length(ufinal_est), 1);
    [err_vec, ind_vec] = min(abs(err_mat));
    SBLaft_doaerrorSqrd_iter(iter, itermultiresSBL+1) = mean(err_vec.^2);
    toc
end
figure
ax=semilogy(1:6,sqrt(mean(SBLbef_doaerrorSqrd_iter(1:iter,:),1)),'--o','Color',[0 0.45 0.74],'LineWidth',1,'MarkerSize',8);
hold on
semilogy(1:6,sqrt(mean(SBLaft_doaerrorSqrd_iter(1:iter,:),1)),'-s','Color',[0 0.45 0.74],'LineWidth',2,'MarkerSize',10)
grid on
semilogy(1:6,sqrcrb_u_theta*ones(1,6),'--k','LineWidth',1)
xlabel('Iteration index')
ylabel('RMSE in u-space')
ax=ax.Parent;
set(ax,'FontWeight','bold','FontSize',14)
xticks([1:6])
legend('Before Peak Adjust.','After Peak Adjust.','CRB')

figure
ax=plot(1:6,mean(GridSize),'-s','LineWidth',2,'MarkerSize',10); hold on
xlabel('Iteration index')
ylabel('Grid Size')
ax=ax.Parent;
set(ax,'FontWeight','bold','FontSize',14)
xticks([1:6])
grid on
plot(2:6,mean(GridSize(:,1:end-1))+K*((5-1)*g+1),'--d','LineWidth',1,'MarkerSize',8)
legend('Actual Grid Size', 'Upper Bound: Based on previous iter. grid size')