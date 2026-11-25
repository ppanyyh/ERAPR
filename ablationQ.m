function [result,labelnew,t,ddd] = ablationQ(X,label,k,mu,eta,lambda,iter)

% remove the updatinig process of Q

if nargin < 7
    iter = 40; 
end
if nargin < 6
    lambda = 1; 
end
if nargin < 5
    eta = 0; 
end
if nargin == 3
    error('The parameters lambda and mu should be input.')
end

tic
pause(1)

%% Initialize
[n,d] = size(X);
c = length(unique(label));   % the number of class

% P
[V_p, D_p] = eig(X' * X);
[~, sort_idx] = sort(diag(D_p), 'descend');
num_to_take = min(c, d); 
P(:, 1:num_to_take) = V_p(:, sort_idx(1:num_to_take));
if c > d
    P(:, d+1:c) = randn(d, c - d) * 1e-4;
end
[Up, ~, Vp] = svd(P, 'econ');
P = Up * Vp';

% Q
Q = eye(c);

% F
init_label = kmeans(X, c);
F = zeros(n, c);
for i = 1:n
    F(i, init_label(i)) = 1;
end

Z = X * P; 
U = zeros(n, c);
beta = 0.01; 

% A
if eta == 0
    [A,mu,~,~] = init_A(X,mu);        
else
    [A,mu,~,~] = init_A(X,mu,eta);    
end
A = (A+A')/2;               
D_a = diag(sum(A,2));       
L_a = D_a - A;              

obj_history = [];

%% Updata
for ddd = 1:iter
    
    % Update P
    P = (X' * X + 1e-6 * eye(d)) \ (X' * (Z - U));
%     P = pinv(X' * X + 1e-3 * eye(d)) * (X' * (Z - U));
    [Up, ~, Vp] = svd(P, 'econ');
    P_temp = Up * Vp'; 
    scale_mat = P_temp' * X' * X * P_temp;
    P = P_temp * (scale_mat ^ -0.5);
    fprintf('normP: %.4e | ', norm(P, 'fro'));
    
    % Upadate Z by GPI
    B_z = beta * (X * P + U) + (F * Q');
    A_z = lambda * L_a + (beta + 1) * eye(n);
    
    % --- GPI 内部迭代 (通常 1-3 次即可) ---
    for gpi_iter = 1:3
        gamma_gpi = max(sum(abs(A_z), 2)); 
        M_gpi = (gamma_gpi * eye(n) - A_z) * Z + B_z;
        [Uz, ~, Vz] = svd(M_gpi, 'econ');
        Z = Uz * Vz';
    end
    
    % Update U
    U = U + (X * P - Z);

    % Update A
    XP = X * P;
    dij_projected = L2_distance_1(XP', XP');
    for i = 1:n
        row_dist = lambda * dij_projected(i, :);
        row_dist(i) = Inf; 
        min_d = min(row_dist);
        row_exp = exp(-(1/mu) * (row_dist - min_d));
        [~, idx] = sort(row_dist, 'ascend'); 
        id_to_zero = idx(k+1:end);           
        row_exp(id_to_zero) = 0;
        sum_i = sum(row_exp);
        if sum_i > 0
            A(i, :) = row_exp / sum_i;
        else
            A(i, idx(1:k)) = 1/k;
        end
    end
    
    A = (A+A')/2;               
    D_a = diag(sum(A,2));       
    L_a = D_a - A;              

    % do not update Q
    % min ||Y - ZQ||^2 s.t. Q'Q = I

    % Update F
    F_old = F;
    Predicted_Labels = Z * Q; 
    F = zeros(n, c);
    [~, cluster_idx] = max(Predicted_Labels, [], 2);
    linear_idx = sub2ind([n, c], (1:n)', cluster_idx);
    F(linear_idx) = 1;

%% Loss Function
    % Graph
    XP_current = X * P;
    dij_p = L2_distance_1(XP_current', XP_current');
    term1 = lambda * sum(sum(dij_p .* A));

    % Entropy
    S_val = A(A > 0);
    term2 = mu * sum(S_val .* log(S_val));

    % Spectral rotation(ablation Q)
    term3 = norm(F - Z * Q, 'fro')^2;

    % ADMM 
    term4 = (beta / 2) * norm(X * P - Z + U, 'fro')^2;
    fprintf('T1:%.4f, T2:%.4f, T3:%.4f, T4:%.4f\n', ...
        full(term1), full(term2), full(term3), full(term4));

    current_obj = term1 + term2 + term3 + term4;
    obj_history(ddd) = current_obj;

    if ddd > 1
        num_changed = sum(abs(F(:) - F_old(:))) / 2;
        change_rate = num_changed / n;

        if change_rate < 0.001 
            fprintf('The algorithm converges at iteration %d. \n', ddd);
            break;
        end
    end

end

[~, labelnew] = max(F, [], 2);
labelnew = labelnew';           
t = toc;
result = zeros(1,5);
result(1:4) = ClusteringMeasure(label,labelnew);
result(5) = t;