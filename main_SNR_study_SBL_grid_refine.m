% Code studies performance of SBL with Likelihood-based Grid Refinement
% when sources are off-grid as a function of SNR
% Code generates Fig. 6 (c) in R. R. Pote and B. D. Rao, "Maximum
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
SNR = -25:5:25; % in dB
P = 1; % power
rho_abs=0; % absolute correlation coefficient
lrho = rand+1i*rand; rho = rho_abs*lrho/abs(lrho);
ITER = 25; % random realizations for averaging
spos = [0 1 2.1 3.5 4.7 10];

% Grid Parameters
sblgrid=150;
u_grid=-1+1/sblgrid:2/sblgrid:1-1/sblgrid;
Agrid=exp(1j*pi*spos'*u_grid);

% Source locations
source_u=(-1+1/K:2/K:1)+(2*rand([1,K])-1)/K/10;
psi = pi*source_u;

% Sources' correlation (uncorrelated if rho_abs=0)
sigma = toeplitz(rho.^(0:K-1));
[Vs, Ds] = eig(sigma);

%% Generating data, run MSBL, run grid refinement step
SBLbef_doaerrorSqrd_iter = zeros(ITER,length(SNR),6); % before peak adjustment
SBLaft_doaerrorSqrd_iter = zeros(ITER,length(SNR),6); % after peak adjustment
GridSize=zeros(ITER,length(SNR),6);
sqrcrb_u_theta=zeros(1,length(SNR));
ITERmultiresSBL=zeros(1,length(SNR));


%% Proposed Algorithm
for iter = 1:ITER
    s = randn(K,L)+1j*randn(K,L);
    n = randn(M,L)+1j*randn(M,L);
    for SNRiter=1:length(SNR)
        tic
        W = P/(10^(SNR(SNRiter)/10)); % noise power
        %% Stochastic CRB equal power sources assumed
        Sigma_s=diag(sqrt(P))*sigma*diag(sqrt(P))';
        num_factor=W/(2*L);
        Acrb = exp(1j*pi*spos'*source_u);
        Dcrb = 1j*spos'.*Acrb;
        Ry_inv = eye(M)/((Acrb*Sigma_s*Acrb')+W*eye(M));
        crb_psi = num_factor*eye(K)/(real(Dcrb'*(eye(M)-((Acrb/(Acrb'*Acrb))*Acrb'))*Dcrb).*((Sigma_s*Acrb'*Ry_inv*Acrb*Sigma_s).'));
        sqrcrb_u_theta(SNRiter)=real(sqrt(mean(diag((1/pi)*crb_psi*((1/pi).')))));
        %% Data Generation
        yo = sqrt(P/2)*exp(1j*spos'*psi)*(Vs*Ds^0.5*Vs')*s(:, 1:L)+sqrt(W/2)*n(:, 1:L);
        Ryoyo = yo*yo'/L;
        %% Run MSBL
        lambda = W;          % Initial value for the regularization parameter.
        learn_Lambda = 0;       % Using its lambda learning rule to learn an (sub-)optimal lambda.
        GridSize(iter,SNRiter,1)=size(Agrid,2);
        [Weight,gamma_ind,gamma_est,count] = MSBL(Agrid,yo, lambda, learn_Lambda,'max_iters', 5000);
        [pks,locs]=findpeaks1(gamma_est);
        [mpks,mlocs]=maxk(pks, K);
        m_candidates=sort(locs(mlocs));% one source
        u_est=u_grid(m_candidates);
        SBLbef_doaerrorSqrd_iter(iter,SNRiter,1)=mean(abs(u_est-source_u).^2);
        
        
        % Parameters
        PRUNE_GAMMA=1e-3;
        g=3;
        resSBL=2/sblgrid;
        u_grid_updated=u_grid;
        Agrid_updated=Agrid;
        itermultiresSBL=0;
        ITERmultiresSBL(SNRiter)=ceil(log(1/sqrcrb_u_theta(SNRiter)/sblgrid)/log(g));
        if ITERmultiresSBL(SNRiter)>0
            %% Perform grid refinement
            for itermultiresSBL=1:ITERmultiresSBL(SNRiter)
                [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,K,u_grid_updated,Agrid_updated,spos,M,Ryoyo,lambda);
                [pks,locs]=findpeaks1(gamma_est);
                [mpks,mlocs]=maxk(pks, K);
                K_est=length(mlocs);
                m_peaks=sort(locs(mlocs));% one source
                u_peaks=u_grid_updated(m_peaks);
                SBLaft_doaerrorSqrd_iter(iter,SNRiter,itermultiresSBL)=mean(abs(u_peaks-source_u).^2);
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
                GridSize(iter,SNRiter,itermultiresSBL+1)=size(Agrid_updated,2);
                [Weight,gamma_ind,gamma_est,count] = MSBL(Agrid_updated,yo, lambda, learn_Lambda,'max_iters', 5000);
                
                [pks,locs]=findpeaks1(gamma_est);
                [mpks,mlocs]=maxk(pks, K);
                K_est=length(mlocs);
                m_peaks=sort(locs(mlocs));% one source
                u_peaks=u_grid_updated(m_peaks);
                SBLbef_doaerrorSqrd_iter(iter,SNRiter,itermultiresSBL+1)=mean(abs(u_peaks-source_u).^2);
            end
        end
        [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,K,u_grid_updated,Agrid_updated,spos,M,Ryoyo,lambda);
        [pks,locs] = findpeaks1(gamma_est);
        [~, mlocs] = maxk(pks, K);
        ufinal_est = sort(u_grid_updated(locs(mlocs)))';
        %%
        err_mat = repmat(ufinal_est, 1, K)-repmat(source_u, length(ufinal_est), 1);
        [err_vec, ind_vec] = min(abs(err_mat));
        SBLaft_doaerrorSqrd_iter(iter,SNRiter,itermultiresSBL+1) = mean(err_vec.^2);
        toc
    end
end
%% Plotting
figure
sqr_SBL_multires=zeros(1,length(SNR));
for iterSNRplot=1:length(SNR)
    if ITERmultiresSBL(iterSNRplot)<0
        sqr_SBL_multires(iterSNRplot)=sqrt(mean(SBLaft_doaerrorSqrd_iter(1:25,iterSNRplot,1),1));
    else
        sqr_SBL_multires(iterSNRplot)=sqrt(mean(SBLaft_doaerrorSqrd_iter(1:25,iterSNRplot,ITERmultiresSBL(iterSNRplot)+1),1));
    end
end
ax=semilogy(SNR,sqr_SBL_multires,'LineWidth',2,'MarkerSize',8);
hold on
grid on
semilogy(SNR,sqrcrb_u_theta,'--k','LineWidth',1)
legend('Proposed Algorithm', 'CRB', 'Location','northeast')
ax = ax.Parent;   % Important
set(ax, 'XTick', SNR); 
set(ax, 'YTick', [1e-5 1e-4 1e-3 1e-2 1e-1 1e0]); 
set(ax, 'FontWeight', 'bold','FontSize',14)
axis([min(SNR) max(SNR) 1e-5 1])
xlabel('SNR in dB', 'FontWeight', 'bold')
ylabel('RMSE in u-space', 'FontWeight', 'bold')