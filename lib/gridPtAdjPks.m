function [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,K,u_grid_updated,Agrid_updated,spos,M,Ryoyo,lambda)
% % Written by Rohan R. Pote, 2022
for iterGdPtAdPks=1:20 % grid point adjustment around peaks iteration
    G=length(gamma_est);
    [pks,locs]=findpeaks1(gamma_est);
    [mpks,mlocs]=maxk(pks, K);
    K_est=length(mlocs);
    m_candidates=sort(locs(mlocs));% one source
    u_est=u_grid_updated(m_candidates);
    for iterK=1:K_est
        m_iterK=m_candidates(iterK);
        if m_iterK>1; left_delta=u_grid_updated(m_iterK)-u_grid_updated(m_iterK-1); else; left_delta=u_grid_updated(m_iterK)+1; end
        if m_iterK<G; right_delta=u_grid_updated(m_iterK+1)-u_grid_updated(m_iterK); else; left_delta=1-u_grid_updated(m_iterK); end
        delta=left_delta/2+right_delta/2; resSeqSBL=delta/G;
        u_candidates=linspace(u_est(iterK)-left_delta/2,u_est(iterK),floor(left_delta/2/resSeqSBL+1));
        u_candidates=union(u_candidates,linspace(u_est(iterK),u_est(iterK)+right_delta/2,floor(right_delta/2/resSeqSBL+1)));
        Gp=length(u_candidates);
        gamma_updated=zeros(1,Gp);
        I_gamma_opt=zeros(1,Gp);
        cmpgdind=setdiff(1:G,m_iterK); % complement set of grid index
        Sigma_mi=Agrid_updated(:,cmpgdind)*diag(gamma_est(cmpgdind))*Agrid_updated(:,cmpgdind)'+lambda*eye(M); % Sigma minus grid point i
        iSigma_mi=eye(M)/Sigma_mi;
        Aadpt_grid=exp(1j*pi*spos'*u_candidates);
        q_i_sq=real(sum(conj(iSigma_mi*Aadpt_grid).*(Ryoyo*iSigma_mi*Aadpt_grid)));
        s_i=real(sum(conj(Aadpt_grid).*(iSigma_mi*Aadpt_grid)));
        q_sq_by_s_i=q_i_sq./s_i;
        gamma_updated(q_i_sq>s_i)=(q_sq_by_s_i(q_i_sq>s_i)-1)./s_i(q_i_sq>s_i);
        I_gamma_opt(q_i_sq>s_i)=log(q_sq_by_s_i(q_i_sq>s_i))-q_sq_by_s_i(q_i_sq>s_i)+1;
        [valneigh,indneigh]=min(I_gamma_opt);
        if valneigh<0
            gamma_est(m_iterK)=gamma_updated(indneigh);
            u_grid_updated(m_iterK)=u_candidates(indneigh);
            Agrid_updated(:,m_iterK)=exp(1j*pi*spos'*u_candidates(indneigh));
        end
    end
end
end