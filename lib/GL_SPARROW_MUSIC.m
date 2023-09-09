function [u_grid,Smusic,u_est]=GL_SPARROW_MUSIC(S,K,spos,mgrid,So,lambda1)
% % Written by Rohan R. Pote, 2022
M=length(spos);
Mapt=spos(end)+1;
x_old = [1 zeros(1,Mapt-1)];

cvx_begin sdp quiet
variable U(M,M) hermitian
variable x(1,Mapt) complex
minimize(real(trace(U*S)+trace(toeplitz(x))/Mapt))
subject to
[U eye(M); eye(M) So*((toeplitz(x)+toeplitz(x)')/2+lambda1*eye(Mapt))*So']>=0
(toeplitz(x)+toeplitz(x)')/2>=0
cvx_end
[u_grid,Smusic,u_est]=root_MUSIC(toeplitz(x),K,0:Mapt-1,mgrid);
end