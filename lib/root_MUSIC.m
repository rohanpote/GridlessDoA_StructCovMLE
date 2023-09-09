function [u_grid,Smusic,u_est]=root_MUSIC(R,K,spos,mgrid)
% % Written by Rohan R. Pote, 2022
% sensor position in spos assumed to start from 0

M=length(spos);
if K<M
    S=R;
    spos_MUSIC=spos;
else % coarray MUSIC
    [xlag, ylag] = meshgrid(spos, spos);
    difflag = ylag-xlag;
    cnsc_lags = min(difflag(:)):max(difflag(:));
    z1 = zeros(length(cnsc_lags), 1);
    for i=1:length(cnsc_lags)
        z1(i, 1) = mean(R(difflag == cnsc_lags(i)));
    end
    l_eps = (length(cnsc_lags)-1)/2;
    Rzzp = zeros(l_eps+1, l_eps+1);
    for j = 1:l_eps+1
        Rzzp = Rzzp+z1(j:j+l_eps)*z1(j:j+l_eps)';
    end
    Rzzp = Rzzp/(l_eps+1);
    S=Rzzp;
    spos_MUSIC=0:l_eps;
end

[V,D] = eig(S);
[~, ind] = sort(diag(D));
En = V(:, ind(1:end-K));
En_expnd = zeros(max(spos_MUSIC)+1,size(En,2));
En_expnd(spos_MUSIC+1,:) = En;
% MUSIC
u_grid=2*(1:mgrid)/mgrid-1;
Phi=exp(1j*pi*(0:max(spos_MUSIC))'*u_grid);
Smusic=1./(vecnorm(En_expnd'*Phi).^2);
% rootMUSIC
B_z = 0;
for ri=1:size(En_expnd, 2)
    %     B_lz = poly2sym(En(:,bb i))*poly2sym(En(:, i)');
    B_lz = conv(En_expnd(:, ri).', fliplr(En_expnd(:, ri)'));
    B_z = B_z+B_lz;
end
ra = roots(B_z);
[dum,ind]=sort(abs(ra));
rb=ra(ind(1:size(En_expnd, 1)-1));

% pick the n roots that are closest to the unit circle
[dumm,I]=sort(abs(abs(rb)-1));
w=angle(rb(I(1:K)));
u_est = sort(w/pi);
end