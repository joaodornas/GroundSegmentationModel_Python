
clear all

% XsXrand = csvread(strcat('XsXRand','.txt'),0,0);
% XXInvrand = csvread(strcat('XXInvRand','.txt'),0,0);
% ProductRand = csvread(strcat('ProductRand','.txt'),0,0);
% 
% ProductRand_mat = XsXrand * XXInvrand;

X = csvread(strcat('X','.txt'),0,0);
Xstar = csvread(strcat('Xstar','.txt'),0,0);
Z = csvread(strcat('Z','.txt'),0,0);

covXX_mat = single(kernel(X,X,1));
covXsX_mat = single(kernel(Xstar,X,0));
covXXs_mat = single(kernel(X,Xstar,0));
covXsXs_mat = single(kernel(Xstar,Xstar,0));
covXXinv_mat = single(inv(covXX_mat));
covInvProduct_mat = single(covXsX_mat*covXXinv_mat);
covProduct_mat = single(covInvProduct_mat*covXXs_mat);
Vstar_mat = single(covXsXs_mat-covProduct_mat);
Fstar_mat = single(covInvProduct_mat*Z);

covXX = csvread(strcat('covXX','.txt'),0,0);
covXXs = csvread(strcat('covXXs','.txt'),0,0);
covXsX = csvread(strcat('covXsX','.txt'),0,0);
covXsXs = csvread(strcat('covXsXs','.txt'),0,0);
covXXinv = csvread(strcat('covXXinv','.txt'),0,0);
covInvProduct = csvread(strcat('covInvProduct','.txt'),0,0);
covProduct = csvread(strcat('covProduct','.txt'),0,0);
Vstar = csvread(strcat('Vstar','.txt'),0,0);
Fstar = csvread(strcat('Fstar','.txt'),0,0);

function cov = kernel(X,Xstar,noise)

    sigma_f = 1.3298;
    sigma_n = 0.1;
    length_scale = 0.3;

    for i = 1:length(X)
        for j = 1:length(Xstar)  
            cov(i,j) = sigma_f * exp( (-1/(2*length_scale^2))*(X(i)-Xstar(j))^2); 
            
            if i==j
                cov(i,j) = cov(i,j) + noise*sigma_n;
            end
        end
    end

end

