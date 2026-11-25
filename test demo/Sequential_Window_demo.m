clear;
% rng(123, 'twister'); 
% DataName = 'CIFAR10';
% DataName = 'Coil20';
% DataName = 'coil100';
% DataName = 'key';
% DataName = 'forest_cover';
% DataName = 'gisettefixed';
% DataName = 'Isolet';
% DataName = 'network';
% DataName = 'ARO';
DataName = 'lung';
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

% --- 超参数设置 ---
window_size = 203;      % 窗口大小（界标）
k = 13; 
alpha = 100;
gamma = 1;
eta = 0.01;

% --- 获取数据总量 ---0.5
% [dim, n_total] = size(X);   % d x n
[n_total, dim] = size(X);   % n x d
num_windows = ceil(n_total / window_size); % 计算总窗口数

%% 窗口模型
start_win = 1;  
end_win = num_windows;    
% process_range = start_win : end_win; % 实际处理的区间 
% fprintf('准备处理窗口区间: %d 到 %d (共 %d 个窗口)\n', start_win, end_win, length(process_range));
% all_results = zeros(length(process_range), 5); 

% 用于存储每个窗口的结果
all_results = zeros(num_windows, 5); % 存储 ACC, NMI, Purity, Time
final_labels = zeros(n_total, 1);    % 存储所有聚类后的标签

fprintf('开始界标窗口处理，总样本数: %d, 窗口大小: %d, 总窗口数: %d\n', n_total, window_size, num_windows);

%% 所有窗口循环处理
for w = 1:num_windows
% for w = process_range
    
    % 1. 确定当前窗口的范围
    start_idx = (w-1) * window_size + 1;
    end_idx = min(w * window_size, n_total);
    fprintf('\n--- 正在处理第 %d 个窗口 (样本范围: %d 到 %d) ---\n', w, start_idx, end_idx);
    
    % 2. 提取当前窗口数据
    
    % d x n
%     X_win = X(:, start_idx:end_idx)'; 
%     L_win = X_label(:, start_idx:end_idx)';

%     n x d
    X_win = X(start_idx:end_idx, :);
    L_win = X_label(start_idx:end_idx, :);

%     PCA 降维
%     [coeff, score, ~] = pca(X_win, 'NumComponents', 1000);
%     X_win = score; % 使用主成0.分得分作为输入特征
    
    % 3. 调用 ERCAN 算法
    % 注意：如果最后一个窗口样本太少，可能需要判断样本数是否大于类别数 c
    try
%         [result, label_win, t_time, ddd_iter] = ERCANN(X_win, L_win, k, alpha, gamma, eta);
        [result, label_win, t_time, ddd_iter] = ERAPR(X_win, L_win, k, mu, eta, lambda);

        % 4. 存储结果
        all_results(w, :) = result;
        final_labels(start_idx:end_idx) = label_win;
%         all_results(w - start_win + 1, :) = result;  %选择窗口

        fprintf('窗口 %d 完成: ACC = %.4f, NMI = %.4f, Purity = %.4f, F1 = %.4f, 耗时 = %.4fs\n', ...
        w, result(1), result(2), result(3), result(4), t_time);
    
    catch ME
        fprintf('窗口 %d 运行出错: %s\n', w, ME.message);
    end
end

%% 汇总分析
avg_result = mean(all_results, 1);

fprintf('\n================汇总结果================\n');
fprintf('平均 ACC:    %.4f\n', avg_result(1));
fprintf('平均 NMI:    %.4f\n', avg_result(2));
fprintf('平均 Purity: %.4f\n', avg_result(3));
fprintf('平均 F1:     %.4f\n', avg_result(4));
fprintf('平均窗口耗时:  %.4fs\n', mean(all_results(:,5)));
fprintf('累计总耗时:   %.4fs\n', sum(all_results(:,5)));
fprintf('========================================\n');

