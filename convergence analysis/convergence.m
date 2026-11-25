%% convergence.m
% 目标函数收敛性分析（左轴：Objective 能量，右轴：聚类精度）
clear;
clc;

% representative_datasets = {'Yale'};
representative_datasets = {'Isolet'};
% representative_datasets = {'Yeast', 'warpAR10P'};
% representative_datasets = {'nci9'};
num_plots = length(representative_datasets);

% 最优超参数配置
% config.Yeast = struct('k', 3, 'mu', 1, 'eta', 1, 'lambda', 1, 'chunk_size', 20);
% config.abalone = struct('k', 4, 'mu', 1, 'eta', 1, 'lambda', 1, 'chunk_size', 120);
% config.ORL = struct('k', 2, 'mu', 1, 'eta', 0.1, 'lambda', 10, 'chunk_size', 20);
% config.AR = struct('k', 3, 'mu', 1, 'eta', 0.001, 'lambda', 10, 'chunk_size', 90);
config.Isolet = struct('k', 3, 'mu', 1, 'eta', 1000, 'lambda', 100, 'chunk_size', 20);
% config.Yale = struct('k', 4, 'mu', 1, 'eta', 0.1, 'lambda', 100, 'chunk_size', 18);
% config.nci9 = struct('k', 2, 'mu', 1, 'eta', 1, 'lambda', 100, 'chunk_size', 10);
% config.warpAR10P = struct('k', 4, 'mu', 1, 'eta', 0.1, 'lambda', 100, 'chunk_size', 10);
max_iter = 30; 

for d_idx = 1:num_plots
    DataName = representative_datasets{d_idx};
    opts = config.(DataName);
    
    try
        load([DataName, '_data.mat']); 
        X = double(X);
        X = X ./ (max(X(:)) + 1e-9); 
        X_label = Y;

        rng(777); % seed
        p = randperm(size(X, 1));
        X = X(p, :);
        X_label = X_label(p, :);
        
        % 截取 Window 1
        start_idx = 1;
        end_idx = 2 * opts.chunk_size; 
        X_win = X(start_idx:end_idx, :);
        L_win = X_label(start_idx:end_idx, :);
        X_win(isnan(X_win)) = 0;

        % PCA
        % [coeff, score, ~] = pca(X_win, 'NumComponents', 500);
        % X_win = score;
        
        % 运行算法并记录历史
        [~, ~, ~, ~, history] = ERAPR_con(X_win, L_win, opts.k, opts.mu, opts.eta, opts.lambda, max_iter);
        
        % 绘图
        fig = figure('Color', 'w', 'Position', [100, 100, 520, 460]);
        iterations = 1:length(history.obj);
        
        % 左轴：目标函数值 J (Objective Value)
        yyaxis left
        p1 = plot(iterations, history.obj, '-d', 'Color', [0.635, 0.078, 0.184], 'LineWidth', 1.8, 'MarkerSize', 5);
        ylabel('Objective Function Value $\mathcal{J}$', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'YColor', [0.635, 0.078, 0.184]); 
        
        % 右轴：聚类精度 (ACC & NMI %)
        yyaxis right
        p2 = plot(iterations, history.acc * 100, '-o', 'Color', [0.85, 0.325, 0.098], 'LineWidth', 1.5, 'MarkerSize', 5);
        hold on;
        p3 = plot(iterations, history.nmi * 100, '-s', 'Color', [0, 0.447, 0.741], 'LineWidth', 1.5, 'MarkerSize', 5);
        ylabel('Clustering Performance (%)', 'FontSize', 12, 'FontWeight', 'bold');
        ylim([0, 105]);
        set(gca, 'YColor', [0.15, 0.15, 0.15]);
        
        % 绘制垂直停止线 (利用 yline 和 xline)
        l1 = xline(history.iter_label_stop, '-.', 'Color', [0.12, 0.53, 0.22], 'LineWidth', 2);
        l2 = xline(history.iter_obj_stop, ':', 'Color', [0.0, 0.45, 0.74], 'LineWidth', 2);
        
        % 细节美化
        grid on;
        xlim([1, max_iter]);
        xticks(1:max_iter);
        xlabel('Number of Iterations', 'FontSize', 12);
        
        % 学术标准子标题
        title(sprintf('(%s) %s', char(96+d_idx), DataName), 'FontSize', 13, 'FontWeight', 'bold');
        
        % --- 【图例美化：优雅地排版在图表正下方】 ---
        legend([p1, p2, p3, l1, l2], ...
               {'Objective $\mathcal{J}$', 'ACC', 'NMI', 'Label-driven Stop (Ours)', 'Obj-driven Stop (Trad.)'}, ...
               'Interpreter', 'latex', ...
               'Location', 'southoutside', ...
               'Orientation', 'horizontal', ...
               'NumColumns', 2, ...  % 排成两行，避免太宽溢出
               'FontSize', 9);
           
        set(gca, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.1);
        
        % --- 【双重导出：一键生成 PNG 和 PDF】 ---
        png_filename = sprintf('%s_convergence.png', DataName);
        pdf_filename = sprintf('%s_convergence.pdf', DataName);
        
        exportgraphics(fig, png_filename, 'Resolution', 300);
        exportgraphics(fig, pdf_filename, 'ContentType', 'vector');
        
        fprintf('数据集 %s 导出成功：%s, %s\n', DataName, png_filename, pdf_filename);
        close(fig); % 自动关闭画布
        
    catch ME
        fprintf('运行出错: %s\n', ME.message);
    end
end
