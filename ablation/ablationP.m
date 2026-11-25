function [result,labelnew,t,ddd] = ablationW(X,label,k,alpha,gamma,eta,iter)

if nargin < 7
    iter = 40; 
end
if nargin < 6
    eta = 0; % 触发 init_S.m 的特殊模式
end
if nargin < 5
    gamma = 1; % 给 gamma 一个默认基础值
end
if nargin < 4
    alpha = 1; % 新增：如果用户没输入 alpha，默认设为 1
end

if nargin == 3
    error('The parameters alpha and gamma should be input.')
end

tic
pause(1)

%% Step1：初始化各变量

[n,d] = size(X);
c = length(unique(label));   % the number of class

% --- 方案二消融实验：W 初始化为类单位矩阵 ---
W = zeros(d, c);
num_identity = min(d, c);
for i = 1:num_identity
    W(i, i) = 1; % 仅在主对角线填充 1
end

% R
R = eye(c);

% Y
init_label = kmeans(X, c);
Y = zeros(n, c);
for i = 1:n
    Y(i, init_label(i)) = 1;
end

Z = X * W; % 辅助变量 Z (n x c)
U = zeros(n, c); % 乘子 U (n x c)
lambda_admm = 0.01; % ADMM 惩罚参数

% S
if eta == 0
    [S,gamma,~,~] = init_S(X,gamma);        % S矩阵和gamma的初始化：S0和gamma0
else
    [S,gamma,~,~] = init_S(X,gamma,eta);    % 时间复杂度为n^{2}*d + n^{2}
end
S = (S+S')/2;               % 对称化
D_s = diag(sum(S,2));       % 度矩阵
L_s = D_s - S;              % 拉普拉斯矩阵

obj_history = [];

% sum_i_original = zeros(n,1);
% dia = zeros(n,1);

%% 变量更新循环

for ddd = 1:iter
    
    % --- 步骤: 更新辅助变量 Z
    B_z = (X * W) + (Y * R');
    A_z = alpha * L_s + 2 * eye(n); % 简化的对齐权重
    [Uz, ~, Vz] = svd(B_z, 'econ');
    Z = Uz * Vz';

    % 步骤2：更新S幂矩阵
    % 计算当前投影空间下的距离矩阵
    XW = X * W;
    dij_projected = L2_distance_1(XW', XW');
    
    for i = 1:n
        % 1. 计算当前行的距离 (alpha * 投影距离)
        % 注意：如果 ERCAN 原始逻辑里还有 Lambda*dij_f，请加上
        row_dist = alpha * dij_projected(i, :);
        row_dist(i) = Inf; % 排除自身
        
        % 2. 【关键】找到当前行最小的距离值
        min_d = min(row_dist);
        
        % 3. 【数值安全更新】减去最小值后计算指数
        % 这样距离最小的点其指数值为 exp(0) = 1，确保分母 >= 1
        row_exp = exp(-(1/gamma) * (row_dist - min_d));
        
        % 4. 排序并执行 L0 稀疏约束 (保留前 k 个邻居)
        [~, idx] = sort(row_dist, 'ascend'); % 按距离升序排
        id_to_zero = idx(k+1:end);           % 除了最近的 k 个，其余全部置零
        row_exp(id_to_zero) = 0;
        
        % 5. 归一化
        sum_i = sum(row_exp);
        if sum_i > 0
            S(i, :) = row_exp / sum_i;
        else
            % 极端防御逻辑：如果万一失效，则在最近邻处平分权重
            S(i, idx(1:k)) = 1/k;
        end
    end
    
    S = (S+S')/2;               % 对称化
    D_s = diag(sum(S,2));       % 度矩阵
    L_s = D_s - S;              % 拉普拉斯矩阵

%% 步骤三：固定S，更新R→Y

    % --- 步骤 3: 更新 R (固定 S, W, Z, Y, U) ---
    % 目标：min ||Y - ZR||^2 s.t. R'R = I
    M = Z' * Y; % 注意：这里用的是辅助变量 Z
    [Ur, ~, Vr] = svd(M, 'econ');
    R = Ur * Vr';

    % --- 步骤 4: 更新 Y (固定 S, W, Z, R, U) ---
    % 目标：寻找 Y 使得 ||Y - ZR||^2 最小
    Y_old = Y;
    Predicted_Labels = Z * R; % 使用谱嵌入 Z 与旋转矩阵 R 的对齐结果
    % Y = zeros(n, c);
    [~, cluster_idx] = max(Predicted_Labels, [], 2); % 找到预测得分最高的列

    % 向量化赋值，避免 for 循环（处理数据流时非常快）
    linear_idx = sub2ind([n, c], (1:n)', cluster_idx);
    Y(linear_idx) = 1;

%% 步骤四 计算损失函数

    % 1. 图项 (保持投影后的局部结构)
    % 注意：此处应使用当前最新的 W 重新计算投影
    XW_current = X * W;
    dij_p = L2_distance_1(XW_current', XW_current');
    term1 = alpha * sum(sum(dij_p .* S));

    % 2. 熵项
    S_val = S(S > 0);
    term2 = gamma * sum(S_val .* log(S_val));

    % 3. 对齐项 (改为 Z 和 Y 的对齐，因为我们是用 Z 更新 R 和 Y 的)
    term3 = norm(Y - Z * R, 'fro')^2;

    % 4. ADMM 惩罚项 (这是保证 XW 逼近 Z 的关键)
    term4 = (lambda_admm / 2) * norm(X * W - Z + U, 'fro')^2;
    % fprintf('T1:%.2f, T2:%.2f, T3:%.2f, T4:%.2f\n', term1, term2, term3, term4);
    % 使用 full() 函数确保输入 fprintf 的是普通数值而非稀疏对象
    fprintf('T1:%.4f, T2:%.4f, T3:%.4f, T4:%.4f\n', ...
        full(term1), full(term2), full(term3), full(term4));

    current_obj = term1 + term2 + term3 + term4;

    obj_history(ddd) = current_obj; % 记录当前损失

    % if ddd > 1
    %     % 计算下降率
    %     relative_change = abs(obj_history(ddd) - obj_history(ddd-1)) / obj_history(ddd-1);
    % 
    %     % 如果变化极小，说明收敛，停止循环
    %     if relative_change < 1e-7
    %         fprintf('算法在第 %d 次迭代收敛。\n', ddd);
    %         fprintf('当前损失: %.4f\n', current_obj);
    %         break;
    %     end
    % end

    if ddd > 1
        % 计算有多少个样本改变了分类
        % 因为 Y 每行只有一个 1，所以 sum(abs(diff))/2 就是改变分类的样本数
        num_changed = sum(abs(Y(:) - Y_old(:))) / 2;
        change_rate = num_changed / n; % 变换率
        
        % 打印一下，方便调试观察
        % fprintf('第 %d 次迭代，分类改变的样本数: %d (%.2f%%)\n', ddd, num_changed, change_rate*100);
        
        % 设定阈值停止：比如只有不到 0.1% 的样本在变动，或者完全不变时停止
        if change_rate < 0.001 
            fprintf('分类已稳定，算法在第 %d 次迭代收敛。\n', ddd);
            break;
        end
    end

end

[~, labelnew] = max(Y, [], 2);

labelnew = labelnew';           % 拉普拉斯矩阵的判定是对应于组成它的相似度矩阵，因此需要判别的矩阵是对称化之后的S矩阵
t = toc;
result = zeros(1,5);
result(1:4) = ClusteringMeasure(label,labelnew);
result(5) = t;