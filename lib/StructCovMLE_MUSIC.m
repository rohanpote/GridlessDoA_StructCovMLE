function [u_grid,Smusic,u_est]=StructCovMLE_MUSIC(S,K,spos,mgrid,So,mmITER,lambda1)
% % Written by Rohan R. Pote, 2022
M=length(spos);
Mapt=spos(end)+1;
x_old = [1 zeros(1,Mapt-1)];
for mmloop = 1:mmITER
    mmloop
    B0 = inv(So*toeplitz(x_old)*So'+lambda1*eye(M));
    cvx_begin sdp quiet
    %     cvx_solver sedumi
    variable x(1,Mapt) complex
    variable U(M,M) hermitian
    minimize(real(trace(B0*(So*toeplitz(x)*So')))+real(trace(U*S)))
    subject to
    (toeplitz(x)+toeplitz(x)')/2>=0
    [U eye(M); eye(M) So*((toeplitz(x)+toeplitz(x)')/2)*So'+lambda1*eye(M)]>=0
    cvx_end
    x_old = x;
end
if K<M
    [u_grid,Smusic,u_est]=root_MUSIC(So*toeplitz(x)*So',K,spos,mgrid);
else
    [u_grid,Smusic,u_est]=root_MUSIC(toeplitz(x),K,0:Mapt-1,mgrid);
end
end