clear;
% DataName = 'CIFAR10';
% DataName = 'Coil20';
% DataName = 'coil100';
% DataName = 'key';
% DataName = 'forest_cover';
% DataName = 'gisettefixed';
DataName = 'warpAR10P';
% DataName = 'network';
% DataName = 'ARO';
% DataName = 'abalonee';
load([DataName,'_data.mat'])

% CIFAR10
% X = double(X);
% X_label = Y;

% coil20
% X = double(X);
% X = zscore(X);
% X = X ./ (max(X(:)) + 1e-9); % 将所有数据映射到 [0, 1] 之间
% X_label = label;

% coil100
% X = double(fea);
% X = zscore(X);
% X = X ./ (max(X(:)) + 1e-9);
% X_label = gnd;

% key
% X = double(key_data);
% X = zscore(X);
% X = X ./ (max(X(:)) + 1e-9); % 将所有数据映射到 [0, 1] 之间
% X_label = key_labels;

% forest_cover
% X = double(forest_cover_data);
% X = X ./ (max(X(:)) + 1e-9);
% X = zscore(X); 
% X_label = forest_cover_labels;

% gisettefixed
X = double(X);
% X = zscore(X);
X = X ./ (max(X(:)) + 1e-9);
X_label = Y;

% Isolet
% X = double(X);
% X = zscore(X); 
% X = X ./ (max(X(:)) + 1e-9);
% X_label = Y;

% network
% X = double(network_data);
% X = X ./ (max(X(:)) + 1e-9);
% X = zscore(X); 
% X_label = network_labels;

% USPS
% X = double(X);
% X = X ./ (max(X(:)) + 1e-9);
% X = zscore(X); 
% X_label = Y;

% AR
% X = double(data_compressed);
% X = X ./ (max(X(:)) + 1e-9);
% X = zscore(X); 
% X_label = labels;

% 在这里打乱数据，确保后续划窗时，每个窗里的类别是均匀的
rng(123); 
p = randperm(size(X, 1));
X = X(p, :);
X_label = X_label(p, :);

%% --- 超参数设置 ---
chunk_size = 20;      % 窗口大小（界标）
k = 4; 
lambda = 100;
mu = 1;
eta = 0.1;

% --- 获取数据总量 ---
% [dim, n_total] = size(X);   % d x n
[n_total, dim] = size(X);   
num_chunks = floor(n_total / chunk_size); 

%% 定义块与窗口
% 窗口1: Chunk 1 + Chunk 2
num_sliding_windows = num_chunks - 1;

% 用于存储每个窗口的结果
all_results = zeros(num_sliding_windows, 5); 
final_labels = zeros(n_total, 1);    % 存储所有聚类后的标签

fprintf('开始滑动窗口处理 (Sliding Window)，总样本: %d, 块大小: %d, 滑动窗口数: %d\n', ...
        n_total, chunk_size, num_sliding_windows);

%% 滑动窗口循环处理
for w = 1:num_sliding_windows
    
    % 1. 确定当前窗口的样本范围
    % 窗口 w 包含当前的块 w 和下一个块 w+1
    start_idx = (w-1) * chunk_size + 1; 
    end_idx = min((w+1) * chunk_size, n_total); % 延伸到下一个块的末尾
    
    fprintf('\n--- 正在处理第 %d 个滑动窗口 (样本范围: %d 到 %d, 共 %d 个样本) ---\n', ...
            w, start_idx, end_idx, end_idx - start_idx + 1);
    
    % 2. 提取当前窗口数据 (包含两个块的重叠数据)

    % d x n
%     X_win = X(:, start_idx:end_idx)'; 
%     L_win = X_label(:, start_idx:end_idx)';

    % n × d
    X_win = X(start_idx:end_idx, :);
    L_win = X_label(start_idx:end_idx, :);
    
    % --- 2. 消融实验：注入噪声特征 ---
%     noise_ratio = 0.05; % 注入 100% 的噪声特征
%     noise_ratio = 0.1;
%     noise_ratio = 0.15;
%     noise_ratio = 0.20;
%     noise_ratio = 0.50;
%     noise_ratio = 1.0;
    noise_ratio = 2.0;
    num_noise_features = floor(size(X_win, 2) * noise_ratio);
    % 生成高斯随机噪声
    noise_features = randn(size(X_win, 1), num_noise_features);
    
    % 拼接原始特征与噪声特征
    X_win_noisy = [X_win, noise_features];
    
    % --- 3. 标准化（非常重要：确保噪声和真特征在同一量级） ---
    X_win_noisy = zscore(X_win_noisy);
    X_win_noisy(isnan(X_win_noisy)) = 0;
    
    %     PCA 降维
%     [coeff, score, ~] = pca(X_win_noisy, 'NumComponents', 500);
%     X_win_noisy = score; 
    
    X_win = X_win_noisy; % 消融实验
    
    %     PCA 降维
%     [coeff, score, ~] = pca(X_win, 'NumComponents', 500);
%     X_win = score; % 使用主成分得分作为输入特征
    
% 3. 调用算法
    try
        % 运行你的聚类算法
        [result, label_win, t_time, ddd_iter] = ERAPR(X_win, L_win, k, mu, eta, lambda);
%         [result, label_win, t_time, ddd_iter] = ERAPR(X_win, L_win, k, mu, eta, lambda);

        % 4. 存储结果
        all_results(w, :) = result;
        final_labels(start_idx:end_idx) = label_win;

        fprintf('窗口 %d 完成: ACC = %.4f, NMI = %.4f, Purity = %.4f, 耗时 = %.4fs\n', ...
        w, result(1), result(2), result(3), t_time);
    
    catch ME
        fprintf('窗口 %d 运行出错: %s\n', w, ME.message);
    end
end

%% 汇总分析
valid_results = all_results(all_results(:,1) > 0, :);
avg_result = mean(all_results, 1);

fprintf('\n================汇总结果================\n');
fprintf('平均 ACC:    %.4f\n', avg_result(1));
fprintf('平均 NMI:    %.4f\n', avg_result(2));
fprintf('平均 Purity: %.4f\n', avg_result(3));
fprintf('平均 F1:     %.4f\n', avg_result(4));
fprintf('平均窗口耗时:  %.4fs\n', mean(all_results(:,5)));
fprintf('累计总耗时:   %.4fs\n', sum(all_results(:,5)));
fprintf('========================================\n');

