function [u_grid,Smusic,u_est]=Fb_SCM_MUSIC(S,K,spos,mgrid)
% % Written by Rohan R. Pote, 2022
M=length(spos);
J = fliplr(eye(M));
S_Fb = (S+J*conj(S)*J)/2; % Forward-Backward averaging
[V,D] = eig(S_Fb);
[~, ind] = sort(diag(D));
En = V(:, ind(1:end-K));
% MUSIC
u_grid=2*(1:mgrid)/mgrid-1;
Phi=exp(1j*pi*spos'*u_grid);
Smusic=1./(vecnorm(En'*Phi).^2);
% root-MUSIC
w = rootmusic1(S, K);
u_est = sort(w/pi);
end