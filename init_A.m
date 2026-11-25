function [A_init,mu,X_exp,sum_i_exp] = init_A(X,mu,eta)
[n,~] = size(X);                    
X_exp = zeros(n,n);                
A_init = zeros(n,n);                
dij_x = L2_distance_1(X',X');       
if nargin == 3
    mu = norm(dij_x,'fro')/n;        
    mu = eta*mu;                
elseif nargin == 1
    error('The parameter gamma should be input before initialization.')
end

%% Initialize A
sum_i_exp = zeros(n,1);          % Complexity = n^{2}
for i = 1:n
    for j = 1:n
        X_exp(i,j) = exp(-(1/mu)*dij_x(i,j));  
    end
    sum_i_exp(i) = sum(X_exp(i,:));                  
    for j = 1:n
        A_init(i,j) = X_exp(i,j)/(sum_i_exp(i));
    end
end