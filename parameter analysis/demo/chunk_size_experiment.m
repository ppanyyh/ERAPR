%% chunk_size_experiment.m
% 用于评估不同块大小对数据流一阶聚类性能的影响
clear;
clc;

%% 1. 数据导入与标准化 (保持与 demo 一致)
DataName = 'nci9';
load([DataName,'_data.mat'])

X = double(X);
X = X ./ (max(X(:)) + 1e-9); 
X_label = Y;

% 设定随机种子以确保实验可重复
rng(123);  
p = randperm(size(X, 1));
X = X(p, :);
X_label = X_label(p, :);

[n_total, dim] = size(X);

%% 2. 候选 chunk_size 参数列表
% candidate_chunks = [20, 40, 80, 120, 160, 200, 300]; % abalone
% candidate_chunks = [10, 15, 20, 25, 30, 35, 40];  % others 
candidate_chunks = [4, 7, 10, 13, 15, 18, 20];  % nci9
% candidate_chunks = [10, 14, 18, 22, 26, 30, 50];  % Yale
% candidate_chunks = [30, 60, 90, 120, 150, 200, 300]; % AR
num_experiments = length(candidate_chunks);

% 预分配空间存储每个 chunk_size 下的平均指标
exp_acc = zeros(num_experiments, 1);
exp_nmi = zeros(num_experiments, 1);
exp_purity = zeros(num_experiments, 1);
exp_time = zeros(num_experiments, 1);

%% 3. 固定超参数设定 (与 demo 一致)
k = 3; 
lambda = 100;
mu = 1;
eta = 1;

fprintf('开始执行 chunk_size 敏感性实验，数据集: %s (总样本: %d)\n', DataName, n_total);
fprintf('------------------------------------------------------------\n');

%% 4. 主循环：遍历不同的 chunk_size
for c_idx = 1:num_experiments
    current_chunk = candidate_chunks(c_idx);
    num_chunks = ceil(n_total / current_chunk); 
    num_sliding_windows = num_chunks - 1;
    
    % 如果窗口数太少，跳过此参数（数据流需要至少2个窗口来体现时序变化）
    if num_sliding_windows < 2
        continue;
    end
    
    fprintf('正在测试 chunk_size = %d (生成 %d 个滑动窗口)... ', current_chunk, num_sliding_windows);
    
    % 用于临时存储当前 chunk_size 下所有窗口的指标
    win_results = zeros(num_sliding_windows, 5); 
    
%     tic; % 开始记录该配置下的总执行时间
    for w = 1:num_sliding_windows
        start_idx = (w-1) * current_chunk + 1;
        end_idx = min((w+1) * current_chunk, n_total);
        
        % n x d
        X_win = X(start_idx:end_idx, :);
        L_win = X_label(start_idx:end_idx, :);
        % d x n
        % X_win = X(:, start_idx:end_idx)';
        % L_win = X_label(:, start_idx:end_idx)';
        
        % PCA
%         [coeff, score, ~] = pca(X_win, 'NumComponents', 500);
%         X_win = score;
        
        try
            % 调用聚类函数
            [result, ~, ~, ~] = ERAPR(X_win, L_win, k, mu, eta, lambda);
            win_results(w, :) = result;
        catch ME
            % 容错处理：若个别窗口计算失败，记录空值
            win_results(w, :) = 0;
        end
    end
%     elapsed_time = toc;
    
    % 过滤掉未成功执行的窗口数据
    valid_indices = win_results(:, 1) > 0;
    if any(valid_indices)
        mean_metrics = mean(win_results(valid_indices, :), 1);
        exp_acc(c_idx) = mean_metrics(1);
        exp_nmi(c_idx) = mean_metrics(2);
        exp_purity(c_idx) = mean_metrics(3);
    else
        exp_acc(c_idx) = 0;
        exp_nmi(c_idx) = 0;
        exp_purity(c_idx) = 0;
    end
    exp_time(c_idx) = sum(win_results(:,5));
    
    fprintf('完成! 平均 ACC: %.4f, 平均 NMI: %.4f, 耗时: %.2fs\n', ...
            exp_acc(c_idx), exp_nmi(c_idx), exp_time(c_idx));
end

%% 5. 打印汇总表格
fprintf('\n================== 实验结果汇总 ==================\n');
fprintf('Chunk_Size\tAvg_ACC\t\tAvg_NMI\t\tTotal_Time(s)\n');
for c_idx = 1:num_experiments
    fprintf('%d\t\t\t%.4f\t\t%.4f\t\t%.2f\n', ...
            candidate_chunks(c_idx), exp_acc(c_idx), exp_nmi(c_idx), exp_time(c_idx));
end
fprintf('==================================================\n');

%% 绘图
% 设定保存的文件名
quality_filename = 'clustering_quality_vs_chunk_size';
time_filename = 'time_cost_vs_chunk_size';

%% 1. 绘制并导出第一个图：聚类精度曲线
fig1 = figure('Position', [100, 100, 480, 380]); % 设定黄金学术比例窗口
plot(candidate_chunks, exp_acc * 100, '-o', 'Color', [0.85, 0.325, 0.098], 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(candidate_chunks, exp_nmi * 100, '-s', 'Color', [0, 0.447, 0.741], 'LineWidth', 2, 'MarkerSize', 6);
grid on;
xticks(candidate_chunks); 
xlabel('Chunk Size (n)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Performance (%)', 'FontSize', 11, 'FontWeight', 'bold');
title('Clustering Quality vs. Chunk Size', 'FontSize', 12);
legend({'Accuracy (ACC)', 'NMI'}, 'Location', 'best');
set(gca, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1);

% 导出为高清晰度 PNG (300 DPI)
print(fig1, quality_filename, '-dpng', '-r300');
% 导出为 PDF 矢量图 (LaTeX 专用，紧凑自适应裁剪)
exportgraphics(fig1, [quality_filename, '.pdf'], 'ContentType', 'vector');

%% 2. 绘制并导出第二个图：计算开销柱状图
fig2 = figure('Position', [150, 150, 480, 380]);
bar(candidate_chunks, exp_time, 0.5, 'FaceColor', [0.466, 0.674, 0.188]);
grid on;
xticks(candidate_chunks); 
xlabel('Chunk Size (n)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Total Computational Cost (seconds)', 'FontSize', 11, 'FontWeight', 'bold');
title('Time Cost vs. Chunk Size', 'FontSize', 12);
set(gca, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1);

% 导出为高清晰度 PNG (300 DPI)
print(fig2, time_filename, '-dpng', '-r300');
% 导出为 PDF 矢量图 (LaTeX 专用，紧凑自适应裁剪)
exportgraphics(fig2, [time_filename, '.pdf'], 'ContentType', 'vector');

fprintf('所有图表已成功导出为高画质 PNG 和 LaTeX 专用 PDF 矢量图！\n');