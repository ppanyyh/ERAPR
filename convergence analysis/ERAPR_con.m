function [result,labelnew,t,ddd, history] = ERAPR_con(X,label,k,mu,eta,lambda,max_iter)

% iter: the maximum number of iterations
% result: 5-dimensional row vector containing ACC, NMI, Purity, F1, and Running Time
% labelnew: Clustering results of the algorithm(n*1)
% r: Rank of Laplacian matrix
% t: Running Time
% ddd: Current number of cycles
% nargin: the number of input parameter
% parameter: 5, mu is given by int_A
% parameter: 4, mu is given by yourself
% parameter: 3, error

% Parameter preprocessing update condition
if nargin < 7
    max_iter = 40; 
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
c = length(unique(label));         % the number of class

% P
% [V, D] = eig(X' * X);
% [~, idx] = sort(diag(D), 'descend');
% P = V(:, idx(1:c)) * diag(1 ./ sqrt(diag(D(idx(1:c), idx(1:c)))));
[V_p, D_p] = eig(X' * X);
[~, sort_idx] = sort(diag(D_p), 'descend');
num_to_take = min(c, d); 
P = zeros(d, c); 
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

% auxiliary variable
Z = X * P; 
U = zeros(n, c); 
beta = 0.01;         % ADMM penalty parameter

% A
if eta == 0
    [A,mu,~,~] = init_A(X,mu);         % A_0 and mu_0
else
    [A,mu,~,~] = init_A(X,mu,eta);         % Complexity: n^{2}*d + n^{2}
end
A = (A+A')/2;               % Symmetrization
D_a = diag(sum(A,2));       % Degree Matrix
L_a = D_a - A;              % Laplacian Matrix 

history.obj = [];
history.acc = [];
history.nmi = [];
history.conn_comp = [];
history.iter_label_stop = []; 
history.iter_obj_stop = [];   

%% Updata
for ddd = 1:max_iter
    % Update P
    % min ||XP - (Z - U)||^2  s.t. P'X'XP = I
    P = (X' * X + 1e-6 * eye(d)) \ (X' * (Z - U));
%     P = pinv(X' * X + 1e-3 * eye(d)) * (X' * (Z - U));

    [Up, ~, Vp] = svd(P, 'econ');
    P_temp = Up * Vp';
    scale_mat = P_temp' * X' * X * P_temp;         % satisfy W'XXW = I
    P = P_temp * (scale_mat ^ -0.5);
    % P = P_temp * ((scale_mat + 1e-9 * eye(size(scale_mat))) ^ -0.5);   % Yeast
    fprintf('normP: %.4e | ', norm(P, 'fro'));        % monitor P
    
    % Upadate Z by GPI
    % Ⅰ. lambda * L_s; Ⅱ. ||F - ZQ||^2 -> -2 * Tr(Z^T * F * Q^T); Ⅲ. beta * ||XP - Z + U||^2
    B_z = beta * (X * P + U) + (F * Q');
    A_z = lambda * L_a + (beta + 1) * eye(n);
    for gpi_iter = 1:3
        % SVD projection of Z = (A - gamma*I)Z - B 
        % The gamma here is the maximum eigenvalue of A_z, used to ensure convergence
        gamma_gpi = max(sum(abs(A_z), 2)); 
        M_gpi = (gamma_gpi * eye(n) - A_z) * Z + B_z;        % Calculate the gradient direction matrix of GPI
        [Uz, ~, Vz] = svd(M_gpi, 'econ');
        Z = Uz * Vz';
    end
    
    % Update U
    U = U + (X * P - Z);

    % Update A
    XP = X * P;
    dij_projected = L2_distance_1(XP', XP');
    for i = 1:n
        
        row_dist = lambda * dij_projected(i, :);        % lambda * projection distance
        row_dist(i) = Inf;        % exclude itself
        
        % Find the minimum distance value for the current row
        % Subtract the minimum value to calculate the index
        % The exponential value of the point with the smallest distance is exp(0) = 1
        % ensuring that the denominator is greater than or equal to >= 1
        min_d = min(row_dist);
        row_exp = exp(-(1/mu) * (row_dist - min_d));
        
        % ||s_i||0 = k
        [~, idx] = sort(row_dist, 'ascend'); % ascending order
        id_to_zero = idx(k+1:end);           % = 0 exclude k nearest neighbors
        row_exp(id_to_zero) = 0;
        
        % normalization
        sum_i = sum(row_exp);
        if sum_i > 0
            A(i, :) = row_exp / sum_i;
        else
            A(i, idx(1:k)) = 1/k; % prevent error: equal division
        end
    end
    
    A = (A+A')/2;               % Symmetrization
    D_a = diag(sum(A,2));       % Degree Matrix
    L_a = D_a - A;              % Laplacian Matrix

    % Update Q
    % min ||F - ZQ||^2  s.t. Q'Q = I
    M = Z' * F; 
    [Ur, ~, Vr] = svd(M, 'econ');
    Q = Ur * Vr';

    % Update F
    % min ||F - ZQ||^2 
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
    A_val = A(A > 0);
    term2 = mu * sum(A_val .* log(A_val));

    % Spectral rotation
    term3 = norm(F - Z * Q, 'fro')^2;

    % ADMM loss in updating P
    term4 = (beta / 2) * norm(X * P - Z + U, 'fro')^2;
    fprintf('T1:%.4f, T2:%.4f, T3:%.4f, T4:%.4f\n', ...
        full(term1), full(term2), full(term3), full(term4));

    current_obj = term1 + term2 + term3 + term4;

    % 2. 提取当前迭代步的离散聚类标签
    [~, current_label] = max(F, [], 2);
    
    % 3. 利用已有的 ClusteringMeasure 函数计算当前精度
    % 假设返回的 metrics 格式为 [ACC, NMI, Purity, F1, ARI]
    metrics = ClusteringMeasure(label, current_label); 
    
    % 4. 存入 history 结构体中 (以当前实际迭代步数 ddd 作为索引)
    history.obj(ddd) = full(current_obj);
    history.acc(ddd) = metrics(1);
    history.nmi(ddd) = metrics(2);
    history.conn_comp(ddd) = length(unique(current_label)); % 记录当前的类别/连通数

    % convergence condition 
    % 1. 记录标签变化率判定点 (is empty 确保只记录第一次触发时的步数)
    if ddd > 1 && isempty(history.iter_label_stop)
        num_changed = sum(abs(F(:) - F_old(:))) / 2;
        change_rate = num_changed / n;
        if change_rate < 0.001
            history.iter_label_stop = ddd; 
        end
    end
    
    % 2. 记录传统目标函数收敛判定点
    if ddd > 1 && isempty(history.iter_obj_stop)
        obj_change = abs(history.obj(ddd) - history.obj(ddd-1)) / abs(history.obj(ddd-1));
        if obj_change < 1e-3  
            history.iter_obj_stop = ddd; 
        end
    end

end
if isempty(history.iter_label_stop); history.iter_label_stop = max_iter; end
if isempty(history.iter_obj_stop); history.iter_obj_stop = max_iter; end

[~, labelnew] = max(F, [], 2);
labelnew = labelnew';         
t = toc;
result = zeros(1,5);
result(1:4) = ClusteringMeasure(label,labelnew);
result(5) = t;