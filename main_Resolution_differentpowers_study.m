% Code compares RAM and proposed StructCovMLE when some sources are 
% closely separated and some have different powers
% Code generates Fig. 5 (e) and (f) in R. R. Pote and B. D. Rao, "Maximum
% Likelihood-Based Gridless DoA Estimation Using Structured Covariance
% Matrix Recovery and SBL With Grid Refinement," in IEEE Transactions on
% Signal Processing, vol. 71, pp. 802-815, 2023.

% % Written by Rohan R. Pote, 2022

clear all; close all
addpath ./lib/
addpath ./lib/RAM_code/

%% Parameters
rng(1)
K = 4; % no. of sources
M = 10; % no. of sensors
L = 1; % no. of snapshots
SNR = 20; % in dB
P = [1/(10^(15/10)) 1 1 0.1]; % source powers
mmITER = 20;
rho_abs=0; lrho = rand+1i*rand; rho = rho_abs*lrho/abs(lrho);
mgrid = 4096; % MUSIC grid size
ITER = 1; % random realizations for averaging
% Sensor positions in Nested array geometry
if mod(M,2)==0
    M1=M/2; M2=M/2;
else
    M1=(M-1)/2; M2=(M+1)/2;
end
spos = union(0:M1-1,M1+(0:M2-1)*(M1+1));
Mapt=spos(end)+1; % Aperture size; also the max. total number of lags
So = eye(Mapt,Mapt); So = So(spos+1, :); % observed samples selection matrix
W = P(2)/(10^(SNR/10)); lambda = W; 
sigma = toeplitz(rho.^(0:K-1));
[Vs, Ds] = eig(sigma);

% Source locations
source_u=[-0.5 -1/Mapt/2 1/Mapt/2 0.6]; %[-1/Mapt/2 1/Mapt/2];% RandomKDoAsPoints(K, sep); %-1+1/K:2/K:1-1/K; %
psi = pi*source_u;
%% Experiment

for iter = 1:ITER
    s = randn(K,L(end))+1j*randn(K,L(end));
    n = randn(Mapt,L(end))+1j*randn(Mapt,L(end));
    for liter=1:length(L)
        y = exp(1j*(0:Mapt-1)'*psi)*(Vs*Ds^0.5*Vs')*diag(sqrt(P/2))*s(:, 1:L(liter))+sqrt(W/2)*n(:, 1:L(liter));
        yo = So*y;
        Ryoyo = yo*yo'/L;
        %% StructCovMLE-Proposed Algorithm
        [u_grid,Smusic,u_est]=StructCovMLE_MUSIC(Ryoyo,K,spos,mgrid,So,mmITER,lambda);
        figure(1)
        ax=plot(u_grid, 10*log10(Smusic/max(Smusic)),'LineWidth', 2); hold on; grid on
        plot(repmat(source_u,length(-80:10:0),1), repmat((-80:10:0)',1,4), '--r', 'LineWidth', 1)
        xlabel('Spatial angle in degrees')
        ylabel('MUSIC pseudospectrum in dB')
        xticks(-1:0.2:1)
        yticks(-80:10:0)
        axis([-1 1 -80 0])
        ax.Parent.GridAlpha=0.5;
        set(ax.Parent,'FontWeight','bold','FontSize',14)
        axes('position',[.185 .65 .25 .25])
        box on
        ax=plot(u_grid(u_grid>=-2/Mapt & u_grid<=2/Mapt), 10*log10(Smusic(u_grid>=-2/Mapt & u_grid<=2/Mapt)/max(Smusic)),'LineWidth',2);
        hold on; grid on
        plot(repmat(source_u(2:3),length(-70:10:0),1), repmat((-70:10:0)',1,2), '--r', 'LineWidth', 1)
        yticks(-50:25:0)
        ax.Parent.GridAlpha=0.5;
        set(ax.Parent,'FontWeight','bold','FontSize',12)
        %% RAM (Courtesy: Prof. Zai Yang)
%         [Q1, ~] = qr(yo',0);
%         yoecon = yo*Q1; Lecon = size(yoecon, 2);
%         n_eta = sqrt(W*(M*L(liter)+2*sqrt(M*L(liter))));
%         [U_all, ~, freq, ~] = RAM_sdpt3(yoecon, spos+1, Mapt, n_eta, K);
%         U=U_all{end};
%         Rhat = (U(Lecon+1:end,Lecon+1:end)+U(Lecon+1:end,Lecon+1:end)')/2;
%         [u_grid,Smusic,~]=root_MUSIC(Rhat,K,0:Mapt-1,mgrid);
%         figure(2)
%         ax=plot(u_grid, 10*log10(Smusic/max(Smusic)),'LineWidth', 2); hold on; grid on
%         plot(repmat(source_u,length(-80:10:0),1), repmat((-80:10:0)',1,4), '--r', 'LineWidth', 1)
%         xlabel('Spatial angle in degrees')
%         ylabel('MUSIC pseudospectrum in dB')
%         xticks(-1:0.2:1)
%         yticks(-80:10:0)
%         axis([-1 1 -80 0])
%         ax.Parent.GridAlpha=0.5;
%         set(ax.Parent,'FontWeight','bold','FontSize',14)
%         axes('position',[.185 .65 .25 .25])
%         box on
%         ax=plot(u_grid(u_grid>=-2/Mapt & u_grid<=2/Mapt), 10*log10(Smusic(u_grid>=-2/Mapt & u_grid<=2/Mapt)/max(Smusic)),'LineWidth',2);
%         hold on; grid on
%         plot(repmat(source_u(2:3),length(-70:10:0),1), repmat((-70:10:0)',1,2), '--r', 'LineWidth', 1)
%         yticks(-50:25:0)
%         ax.Parent.GridAlpha=0.5;
%         set(ax.Parent,'FontWeight','bold','FontSize',12)
    end
end