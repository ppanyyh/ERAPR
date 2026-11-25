function [result,labelnew,t,ddd] = ERCANN(X,label,k,alpha,gamma,eta,iter)
% ERCANN是关于W更新的简化ADMM+Procrutes投影算法
% X为原始数据n*d
% label为真实标签向量n*1，用于验证ERCAN聚类结果的聚类指标
% 该方法利用||s_i||0 = k对相似度矩阵进行强稀疏约束，k为图近邻数
% 如果输入为5个，默认gamma在init_S中由F范数确定，sigma用来调整倍率
% 如果输入为4个，默认gamma人为给定，sigma的值为0
% 如果输入为3个，会提醒报错，未给定gamma值
% iter为最大迭代次数

% result: 包含ACC,NMI,Purity和Running Time的4维行向量
% labelnew: ERCAN算法聚类结果n*1
% r: Rank of Laplacian matrix
% t: Running Time
% ddd: 当前循环次数
% % self_similar 0：每个样本自身与自身的相似度被限定为0；非0：每个样本自身与自身存在相似度
% % k0范数约束参数：||s_i||0 = k,若self_similar = 1,则||s_i||0 = k+1
% % nargin 是一个内置变量，它会告诉你用户实际传入了多少个参数

% --- 参数预处理逻辑更新 ---
self_similar = 0;
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

% W
[V, D] = eig(X' * X);
[~, idx] = sort(diag(D), 'descend');
W = V(:, idx(1:c)) * diag(1 ./ sqrt(diag(D(idx(1:c), idx(1:c)))));

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

% dij_x = L2_distance_1(X',X');
% gamma = norm(dij_x,'fro')/n;
% gamma = norm(dij_projected,'fro')/n;        % 利用F范数进行预初始化gamma 这一步很耗时，看怎么优化
% gamma = eta*gamma;
% S = zeros(n, n);

% S
if eta == 0
    [S,gamma,~,~] = init_S(X,gamma);        % S矩阵和gamma的初始化：S0和gamma0
else
    [S,gamma,~,~] = init_S(X,gamma,eta);    % 时间复杂度为n^{2}*d + n^{2}
end
S = (S+S')/2;               % 对称化

obj_history = [];

%% 变量更新循环

for ddd = 1:iter

    % --- 步骤 1: 更新 W (固定 S, R, Y, Z, U) ---
    % 根据 SR-OSC 论文公式 (11)，W 有闭式解
    % W = (X'X)^-1 * X' * (Z + U)
    % W = (X' * X + 1e-6*eye(d)) \ (X' * (Z - U));
    % 将正则化参数从 1e-6 增加到 0.1 或 1
    XX_inv = (X' * X + 1e-6 * eye(d));
    W = XX_inv \ (X' * (Z - U));
    fprintf('normW: %d\n' , norm(W));

    % --- 步骤 1b: 更新辅助变量 Z (固定 S, W, R, Y, U) ---
    % Z 受到图项和对齐项的共同作用，且满足 Z'Z = I
    % 构造 GPI 求解矩阵 A_z
    A_z = lambda_admm * (X * W + U) + (Y * R'); % 这里维度是 (n x c)
    % 根据论文公式 (13) 及其定理，这里需要求解一个正交约束问题
    [Uz, ~, Vz] = svd(A_z, 'econ');
    Z = Uz * Vz';
    
    % --- 步骤 1c: 更新乘子 U ---
    U = U + (X * W - Z);
    
      % 新方案
%     % --- 步骤 1a: 更新 W (固定 S, R, Y, Z, U) ---
%     % 目标：min ||XW - (Z - U)||^2  (注意：这里不加正交约束，正交由 Z 保证)
%     % 岭回归参数正则化：le-6 足够，不要用 10，否则会把 W 压扁
%     W = (X' * X + 1e-6 * eye(d)) \ (X' * (Z - U));
%     
%     % 2. 【核心修改】强制投影到 W'X'XW = I 约束空间
%     % 这能防止 normW 飙升，保证数值稳定性
%     [Uw, ~, Vw] = svd(W, 'econ');
%     W_temp = Uw * Vw'; % 保持正交基
%     % 缩放以满足 W'XXW = I
%     scale_mat = W_temp' * X' * X * W_temp;
%     W = W_temp * (scale_mat ^ -0.5);
%     
%     % 监控 normW
%     fprintf('normW: %.4e | ', norm(W, 'fro'));
%     
%     % --- 步骤 1b: 更新辅助变量 Z (固定 S, W, R, Y, U) ---
%     % 根据 SR-OSC 论文公式 (12)-(13)，Z 受到三方拉扯：
%     % 1. 图正则项: alpha * L_s
%     % 2. 标签项: ||Y - ZR||^2 -> -2 * Tr(Z^T * Y * R^T)
%     % 3. ADMM一致性项: lambda_admm * ||XW - Z + U||^2
%     
%     % 构造 GPI (广义幂迭代) 的线性项矩阵 B (参考 SR-OSC Eq.13)
%     % 这里 lambda_admm 建议设为 1 或更高，以强迫 XW = Z
%     B_z = lambda_admm * (X * W + U) + (Y * R');
%     
%     % 构造 GPI 的二次项矩阵 A (参考 SR-OSC Eq.13)
%     % A 决定了谱空间的聚集程度
%     A_z = alpha * L_s + (lambda_admm + 1) * eye(n);
%     
%     % --- GPI 内部迭代 (通常 1-3 次即可) ---
%     for gpi_iter = 1:3
%         % 这是 GPI 的核心步：Z = (A - gamma*I)Z - B 的 SVD 投影
%         % 这里的 gamma 是 A_z 的最大特征值，用于保证收敛
%         gamma_gpi = max(sum(abs(A_z), 2)); % 极大的特征值估计
%         
%         % 计算 GPI 的梯度方向矩阵
%         M_gpi = (gamma_gpi * eye(n) - A_z) * Z + B_z;
%         
%         % 通过 SVD 进行正交化投影
%         [Uz, ~, Vz] = svd(M_gpi, 'econ');
%         Z = Uz * Vz';
%     end
%     
%     % --- 步骤 1c: 更新乘子 U ---
%     U = U + (X * W - Z);

    % 步骤2：更新S幂矩阵
    % 计算当前投影空间下的距离矩阵
    XW = X * W;
    dij_projected = L2_distance_1(XW', XW');
    dij = zeros(n,n);                                                 % dij初始
    D_exp_update = zeros(n,n);
    sum_i_original = zeros(n,1);
    
    for i = 1:n
        for j = 1:n
            % dij(i,j) = dij_x(i,j) + Lambda*dij_f(i,j);
            dij(i,j) = alpha * dij_projected(i,j);% 投影空间下的dij
            D_exp_update(i,j) = exp(-(1/gamma)*dij(i,j));             % 求幂矩阵
        end
%         dia(i) = D_exp_update(i,i);
        sum_i_original(i) = sum(D_exp_update(i,:));
        [~,idx] = sort(D_exp_update(i,:),'descend');                  % 从大到小距离排序：对指数项降序排序，等价于对距离 dij 升序排序，高效地找到了最近的邻居。
        id = idx(k+2:n);                                              % 选出其余k+1到n的距离元素
        D_exp_update(i,id) = 0;                                       % 让比较大距离的元素全部为0
        if self_similar == 0
            D_exp_update(i,i) = 0;
        end
        sum_i_update = sum(D_exp_update(i,:));                        % 该第i行k个元素的行和
        for j = 1:n
            S(i,j) = D_exp_update(i,j)/(sum_i_update);                % 求每个元素对行和的占比：sij的公式推导结果
        end
    end
    
% 新计算S的方案    
%     for i = 1:n
%         1. 计算当前行的距离 (alpha * 投影距离)
%         注意：如果 ERCAN 原始逻辑里还有 Lambda*dij_f，请加上
%         row_dist = alpha * dij_projected(i, :);
%         row_dist(i) = Inf; % 排除自身
%         
%         2. 【关键】找到当前行最小的距离值
%         min_d = min(row_dist);
%         
%         3. 【数值安全更新】减去最小值后计算指数
%         这样距离最小的点其指数值为 exp(0) = 1，确保分母 >= 1
%         row_exp = exp(-(1/gamma) * (row_dist - min_d));
%         
%         4. 排序并执行 L0 稀疏约束 (保留前 k 个邻居)
%         [~, idx] = sort(row_dist, 'ascend'); % 按距离升序排
%         id_to_zero = idx(k+1:end);           % 除了最近的 k 个，其余全部置零
%         row_exp(id_to_zero) = 0;
%         
%         5. 归一化
%         sum_i = sum(row_exp);
%         if sum_i > 0
%             S(i, :) = row_exp / sum_i;
%         else
%             极端防御逻辑：如果万一失效，则在最近邻处平分权重
%             S(i, idx(1:k)) = 1/k;
%         end
%     end
    
    S = (S+S')/2;               % 对称化
    

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