clear; clc;
%% 数据加载与预处理
DataName = 'warpAR10P';
load([DataName,'_data.mat'])
X_orig = double(X);
X_orig = X_orig ./ (max(X_orig(:)) + 1e-9);
X_label_orig = Y;

% 实验基础配置
chunk_size = 10; 
mu = 1;

% 最佳基准参数
best_k = 4;
best_eta = 0.1;
best_lambda = 100;

%% 窗口一致性测试 
% 使用最优参数，展示每一个窗口的波动情况
fprintf('\n开始执行窗口一致性测试...\n');
rng(2026); % 固定种子
p = randperm(size(X_orig, 1));
X_test = X_orig(p, :);
L_test = X_label_orig(p, :);

[n_total, dim] = size(X_test);
num_chunks = ceil(n_total / chunk_size);
num_windows = num_chunks - 1;
window_accs = zeros(num_windows, 1);
for w = 1:num_windows
    idx = (w-1)*chunk_size+1 : min((w+1)*chunk_size, n_total);
    X_win = X_test(idx, :);
    L_win = L_test(idx, :);
    
    % PCA 依据数据确认是否开启
    % [coeff, score, ~] = pca(X_win, 'NumComponents', 500);
    % X_win = score; % 使用主成分得分作为输入特征
    
    [res, ~, ~, ~] = ERAPR(X_win, L_win, best_k, mu, best_eta, best_lambda);
    window_accs(w) = res(1);
end

%% --- 绘图与保存 ---

fig = figure('Color', 'w', 'Position', [100 100 800 450]); % 稍微拉宽，更显专业

% 1. 自动计算更合适的纵轴范围
y_min = min(window_accs);
y_max = max(window_accs);
y_range = [max(0, y_min - 0.1), min(1, y_max + 0.15)]; % 动态留白

% 2. 绘制原始数据（用浅蓝色，降低视觉压力）
p1 = plot(window_accs, 'Color', [0.8, 0.8, 1], 'LineWidth', 0.8, 'HandleVisibility', 'on'); 
hold on;

% 3. 增加平滑趋势线（核心：展示稳定性趋势）
% 使用 5 窗口移动平均
smoothed_accs = smoothdata(window_accs, 'movmean', 5);
p2 = plot(smoothed_accs, 'b', 'LineWidth', 2); 

% 4. 绘制平均值基准线（使用细虚线）
mean_val = mean(window_accs);
p3 = plot([1, num_windows], [mean_val, mean_val], 'r--', 'LineWidth', 1.5);

% 5. 坐标轴与范围调整
xlim([1, num_windows]);
ylim(y_range);
if num_windows <= 15
    % 如果窗口较少，显示每一个整数索引
    set(gca, 'XTick', 1:num_windows); 
else
    % 如果窗口较多（如超过15个），为了美观，每隔 2 或 5 个显示一个整数刻度
    % 这样可以防止刻度文字重叠
    step = ceil(num_windows / 10); % 自动计算步长，保证刻度约有10个左右
    set(gca, 'XTick', 1:step:num_windows); 
end
grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5); % 虚化网格，突出主体

% 6. 专业标签与字体
title('Dynamic Stability of Clustering Performance', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Window Index (Temporal Evolution)', 'FontSize', 12);
ylabel('Accuracy (ACC)', 'FontSize', 12);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

% 7. 优化图例
legend([p1, p2, p3], {'Raw Window ACC', 'Smoothed Trend', ['Mean (', num2str(mean_val, '%.4f'), ')']}, ...
       'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off');

% 8. 增加一点美感：在均值线上下填充浅色区域（可选，展示波动区间）
fill_x = [1:num_windows, fliplr(1:num_windows)];
fill_y = [ones(1,num_windows)*(mean_val+std(window_accs)), ones(1,num_windows)*(mean_val-std(window_accs))];
% fill(fill_x, fill_y, 'r', 'FaceAlpha', 0.05, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% 导出
exportgraphics(gca, 'Stability_Refined.png', 'Resolution', 600);
exportgraphics(gca, 'Stability_Refined.pdf', 'ContentType', 'vector');