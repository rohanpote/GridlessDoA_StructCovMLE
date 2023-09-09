function [u_grid,Smusic,u_est]=GL_SPICE_MUSIC(S,K,spos,mgrid,So,L)
% % Written by Rohan R. Pote, 2022
M=length(spos);
Mapt=spos(end)+1;
x_old = [1 zeros(1,Mapt-1)];

if L>=M
    [V, D] = eig(S);
    Rh = V*D^(0.5)*V';
    iR = V*D^(-1)*V';
    cvx_begin sdp quiet
    variable X(M,M) hermitian
    variable x(1,Mapt) complex
    minimize(real(trace(X)+trace(So'*iR*So*(toeplitz(x)+toeplitz(x)')/2)))
    subject to
    (toeplitz(x)+toeplitz(x)')/2>=0
    [X Rh; Rh' So*((toeplitz(x)+toeplitz(x)')/2)*So']>=0
    cvx_end
else
    cvx_begin sdp quiet
    variable X(M,M) hermitian
    variable x(1,Mapt) complex
    minimize(real(trace(X)+trace(So'*So*toeplitz(x))))
    subject to
    (toeplitz(x)+toeplitz(x)')/2>=0
    [X S; S So*((toeplitz(x)+toeplitz(x)')/2)*So']>=0
    cvx_end
end
[u_grid,Smusic,u_est]=root_MUSIC(toeplitz(x),K,0:Mapt-1,mgrid);
end