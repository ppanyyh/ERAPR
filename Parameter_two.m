clear; clc;
%% 数据加载与预处理
DataName = 'nci9';
load([DataName,'_data.mat'])
X_orig = double(X);
X_orig = X_orig ./ (max(X_orig(:)) + 1e-9);
X_label_orig = Y;

% 实验基础配置
test_seeds = [123, 42, 1337]; 
chunk_size = 10; 
mu = 1;

% 最佳基准参数
best_k = 3;
best_eta = 1;
best_lambda = 100;

% Isolet
% range_lambda = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000];
% range_eta   = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000];
% range_k     = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

% abalone
% range_lambda = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000];
% range_eta   = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000];
% range_k     = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

% nci9
range_lambda = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000];
range_eta   = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000];
range_k     = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

%% 双变量协同分析
% lambda vs eta
fprintf('\n开始生成 lambda vs eta 热力图数据...\n');
heatmap_data = zeros(length(range_eta), length(range_lambda));
for i = 1:length(range_eta)
    for j = 1:length(range_lambda)
        [m_acc, ~] = run_multi_seed_eval(X_orig, X_label_orig, 123, ... % 热力图通常用1个种子加快速度
                        chunk_size, best_k, range_lambda(j), mu, range_eta(i));
        heatmap_data(i, j) = m_acc;
    end
end

% lambda vs k 
fprintf('\n开始生成 lambda vs k 热力图数据...\n');
heatmap_alpha_k = zeros(length(range_k), length(range_lambda));

for i = 1:length(range_k)
    for j = 1:length(range_lambda)
        [m_acc, ~] = run_multi_seed_eval(X_orig, X_label_orig, 123, ...
                        chunk_size, range_k(i), range_lambda(j), mu, best_eta);
        heatmap_alpha_k(i, j) = m_acc;
    end
end

%% --- 3. 绘图部分：3D 协同分析 ---
figure('Color', 'w', 'Position', [100, 100, 1100, 500]);

% --- 子图1: lambda vs eta ---
ax1 = subplot(1, 2, 1);
% 直接调用下方定义的函数
draw_3d_synergy(heatmap_data, range_lambda, range_eta, '\lambda', '\eta', 'Synergy: \lambda vs \eta');

% --- 子图2: lambda vs k ---
ax2 = subplot(1, 2, 2);
draw_3d_synergy(heatmap_alpha_k, range_lambda, range_k, '\lambda', 'k', 'Synergy: \lambda vs k');

%% --- 4. 自动保存 ---
set(gcf, 'InvertHardcopy', 'off');
exportgraphics(ax1, '3D_lambda_Eta.png', 'Resolution', 600);
exportgraphics(ax1, '3D_lambda_Eta.pdf', 'ContentType', 'vector');
exportgraphics(ax2, '3D_lambda_K.png', 'Resolution', 600);
exportgraphics(ax2, '3D_lambda_K.pdf', 'ContentType', 'vector');

function draw_3d_synergy(Z, x_range, y_range, x_label_str, y_label_str, title_str)
    % 1. 绘制3D柱状图
    h = bar3(Z);
    
    % 2. 实现“高度上色”的核心逻辑
    for i = 1:length(h)
        zdata = get(h(i), 'ZData');
        set(h(i), 'CData', zdata, 'FaceColor', 'interp');
    end
    
    colormap jet; colorbar;
    
    % --- 核心修改：针对 X 轴生成 10^{n} 标签 ---
    if contains(x_label_str, 'lambda') || contains(x_label_str, 'eta')
        x_labels = cell(1, length(x_range));
        for i = 1:length(x_range)
            exponent = round(log10(x_range(i)));
            x_labels{i} = ['10^{', num2str(exponent), '}'];
        end
        set(gca, 'XTick', 1:length(x_range), 'XTickLabel', x_labels, 'TickLabelInterpreter', 'tex');
    else
        set(gca, 'XTick', 1:length(x_range), 'XTickLabel', string(x_range));
    end

    % --- 核心修改：针对 Y 轴生成 10^{n} 标签 ---
    if contains(y_label_str, 'lambda') || contains(y_label_str, 'eta')
        y_labels = cell(1, length(y_range));
        for i = 1:length(y_range)
            exponent = round(log10(y_range(i)));
            y_labels{i} = ['10^{', num2str(exponent), '}'];
        end
        set(gca, 'YTick', 1:length(y_range), 'YTickLabel', y_labels, 'TickLabelInterpreter', 'tex');
    else
        set(gca, 'YTick', 1:length(y_range), 'YTickLabel', string(y_range));
    end

    % 3. 设置标签（启用 Interpreter 为 tex 以支持希腊字母）
    xlabel(x_label_str, 'FontWeight', 'bold', 'Interpreter', 'tex');
    ylabel(y_label_str, 'FontWeight', 'bold', 'Interpreter', 'tex');
    zlabel('Accuracy (ACC)', 'FontWeight', 'bold');
    title(title_str, 'FontSize', 12, 'Interpreter', 'tex');

    % 4. 视角与字体优化
    view(-30, 30);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
    grid on;
end