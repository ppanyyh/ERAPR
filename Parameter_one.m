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

%% 单变量敏感性分析
% lambda
% fprintf('正在计算 lambda 敏感性...\n');
% for i = 1:length(range_lambda)
    % [m_acc, m_nmi] = run_multi_seed_eval(X_orig, X_label_orig, test_seeds, chunk_size, best_k, range_lambda(i), mu, best_eta);
    % data_lambda(i, :) = [m_acc, m_nmi];
% end
% plot_and_save_style(data_lambda, range_lambda, 'parameter \lambda', 'Sensitivity_lambda');

% eta
% fprintf('正在计算 Eta 敏感性...\n');
% for i = 1:length(range_eta)
    % [m_acc, m_nmi] = run_multi_seed_eval(X_orig, X_label_orig, test_seeds, chunk_size, best_k, best_lambda, mu, range_eta(i));
    % data_eta(i, :) = [m_acc, m_nmi];
% end
% plot_and_save_style(data_eta, range_eta, 'parameter \eta', 'Sensitivity_Eta');

% k
fprintf('正在计算 K 敏感性...\n');
for i = 1:length(range_k)
    [m_acc, m_nmi] = run_multi_seed_eval(X_orig, X_label_orig, test_seeds, chunk_size, range_k(i), best_lambda, mu, best_eta);
    data_k(i, :) = [m_acc, m_nmi];
end
plot_and_save_style(data_k, range_k, 'parameter k', 'Sensitivity_K');

%% 绘图
function plot_and_save_style(data, range_vals, x_label_str, file_name)
    % 创建画布
    fig = figure('Units', 'pixels', 'Position', [100 100 500 400], 'Visible', 'on');
    hold on; box on;
    
    % 设置 X 轴为等间距刻度
    x_ticks = 1:length(range_vals);
    
    % 绘制 ACC 和 NMI
    p1 = plot(x_ticks, data(:,1)*100, '-*r', 'LineWidth', 1.2, 'MarkerSize', 6); 
    p2 = plot(x_ticks, data(:,2)*100, '-ob', 'LineWidth', 1.2, 'MarkerSize', 6);
    xlim([1, length(range_vals)]);
    
    % 只有当参数跨度很大（如 lambda 和 eta）时才使用幂次格式
    if contains(x_label_str, 'lambda') || contains(x_label_str, 'eta')
        new_labels = cell(1, length(range_vals));
        for i = 1:length(range_vals)
            exponent = round(log10(range_vals(i))); % 获取 10 的幂次
            new_labels{i} = ['10^{', num2str(exponent), '}']; % 构造 TeX 格式字符串
        end
        xticks(x_ticks);
        % 设置刻度并启用 Tex 解析器
        set(gca, 'xticklabel', new_labels, 'TickLabelInterpreter', 'tex'); 
    else
        % 对于 k 这种线性增长的参数，保持原样
        xticks(x_ticks);
        xticklabels(string(range_vals));
    end
    % ---------------------------------------

    % 设置坐标轴标签
    xlabel(x_label_str, 'FontName', 'Times New Roman', 'FontSize', 14, 'Interpreter', 'tex');
    ylabel('Performance (%)', 'FontName', 'Times New Roman', 'FontSize', 14);
    
    % 设置整体样式
    ylim([50 70]); 
    set(gca, 'FontSize', 12, 'FontName', 'Times New Roman', 'LineWidth', 1);
    grid on; % 建议开启网格，看起来更专业
    
    % 图例
    legend([p1, p2], {'ACC', 'NMI'}, 'Location', 'SouthEast', ...
           'FontName', 'Times New Roman', 'FontSize', 12);
    
    % 保存图片
    exportgraphics(gca, [file_name, '.png'], 'Resolution', 300);
    exportgraphics(gca, [file_name, '.pdf'], 'ContentType', 'vector');

end