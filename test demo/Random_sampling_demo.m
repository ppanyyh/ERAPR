% DataName = 'CIFAR10';
% DataName = 'Coil20';
% DataName = 'coil100';
% DataName = 'key';
% DataName = 'forest_cover';
% DataName = 'gisettefixed';
% DataName = 'glass';
% DataName = 'Isolet';
DataName = 'network';
% DataName = 'USPS';
% DataName = 'ORL';
load([DataName,'_data.mat'])

% CIFAR10
% X_label = Y;

% coil20
% X = double(X);
% X = zscore(X);
% % X = X ./ (max(X(:)) + 1e-9); % 将所有数据映射到 [0, 1] 之间
% X_label = label;

% coil100
% X = double(fea);
% X = zscore(X);
% % X = X ./ (max(X(:)) + 1e-9);
% X_label = gnd;

% key
% X = double(key_data);
% X = zscore(X);
% % % X = X ./ (max(X(:)) + 1e-9); % 将所有数据映射到 [0, 1] 之间
% X_label = key_labels;

% forest_cover
% X = double(forest_cover_data);
% % X = X ./ (max(X(:)) + 1e-9);
% X = zscore(X); 
% X_label = forest_cover_labels;

% gisettefixed
% X = double(X);
% X = zscore(X);
% % X = X ./ (max(X(:)) + 1e-9);
% X_label = Y;

% glass
% X = double(X);
% X = zscore(X);
% % X = X ./ (max(X(:)) + 1e-9);
% X_label = Y;

% Isolet
% X = double(X);
% X = zscore(X); 
% X = X ./ (max(X(:)) + 1e-9);
% X_label = Y;

% network
X = double(network_data);
% X = X ./ (max(X(:)) + 1e-9);
X = zscore(X); 
X_label = network_labels;

% USPS
% X = double(X);
% X = X ./ (max(X(:)) + 1e-9);
% X_label = Y;

% Yeast
% X = double(X);
% X = zscore(X);
% X = X ./ (max(X(:)) + 1e-9);
% X_label = Y;


% --- 超参数设置 ---
sample_size = 200;      % 每次实验采样的样本量
num_experiments = 10;     % 重复实验次数，用于观察稳定性
k = 5; 
alpha = 1000;
gamma = 1;
eta = 0.01;

% --- 获取数据总量 ---
[dim, n_total] = size(X);   % d x n
% [n_total, dim] = size(X);   % n x d

%% 选择某几个窗口
% start_win = 11;  % 你想查看的起始窗口编号
% end_win = 15;    % 你想查看的结束窗口编号
% process_range = start_win : end_win; % 实际处理的区间 
% fprintf('准备处理窗口区间: %d 到 %d (共 %d 个窗口)\n', start_win, end_win, length(process_range));
% all_results = zeros(length(process_range), 4); 

% 用于存储每次采样的结果
all_results = zeros(num_experiments, 5); 
final_labels = zeros(n_total, 1);    

fprintf('开始随机采样实验，总样本池: %d, 每次采样: %d, 实验次数: %d\n', n_total, sample_size, num_experiments);

%% 随机采样实验循环
for e = 1:num_experiments
    % for w = process_range
    
    fprintf('\n--- 正在执行第 %d 次随机采样实验 ---\n', e);
    
    % 1. 生成随机不重复索引
    % randperm(n, k) 会从 1 到 n 之间随机选出 k 个不重复的整数
    sample_indices = randperm(n_total, sample_size);

    % 2. 提取数据
    
    % d x n
    X_win = X(:, sample_indices)'; 
    L_win = X_label(:, sample_indices)';

%     n x d
%     X_win = X(sample_indices, :);
%     L_win = X_label(sample_indices, :);

    % 3. 调用 ERCAN 算法
    % 注意：如果最后一个窗口样本太少，可能需要判断样本数是否大于类别数 c
    try
        [result, label_win, t_time, ddd_iter] = ERAPR(X_win, L_win, k, mu, eta, lambda);
        
        % 4. 存储结果
        all_results(e, :) = result;
        final_labels(sample_indices) = label_win;
%         all_results(w - start_win + 1, :) = result;  %选择窗口

        fprintf('实验 %d 完成: ACC = %.4f, NMI = %.4f, Purity = %.4f, F1 = %.4f, 耗时 = %.4fs\n', ...
        e, result(1), result(2), result(3), result(4), t_time);
    
    catch ME
        fprintf('实验 %d 运行出错: %s\n', e, ME.message);
    end
end

%% 汇总分析
avg_result = mean(all_results, 1);

fprintf('\n================汇总结果================\n');
fprintf('平均 ACC:    %.4f\n', avg_result(1));
fprintf('平均 NMI:    %.4f\n', avg_result(2));
fprintf('平均 Purity: %.4f\n', avg_result(3));
fprintf('平均 F1:     %.4f\n', avg_result(4));
fprintf('平均实验耗时:  %.4fs\n', mean(all_results(:,5)));
fprintf('累计总耗时:   %.4fs\n', sum(all_results(:,5)));
fprintf('========================================\n');
