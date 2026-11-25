clear;

%% data input

DataName = 'nci9';
load([DataName,'_data.mat'])

X = double(X);
X = X ./ (max(X(:)) + 1e-9);
X_label = Y;

rng(123);  % seed
p = randperm(size(X, 1));
X = X(p, :);
X_label = X_label(p, :);

%% parameter settings 

chunk_size = 15;      % chunk size for window
k = 3; 
lambda = 100;
mu = 1;
eta = 1;

%% define window

% get total data 
% [dim, n_total] = size(X);   % d x n
[n_total, dim] = size(X);   
num_chunks = ceil(n_total / chunk_size); 

% window_1 = chunk_1 + chunk_2
num_sliding_windows = num_chunks - 1;

% algorithm results
all_results = zeros(num_sliding_windows, 5); 
final_labels = zeros(n_total, 1);   

fprintf('total data: %d, chunk size: %d, windows count: %d\n', ...
        n_total, chunk_size, num_sliding_windows);

for w = 1:num_sliding_windows
    
    start_idx = (w-1) * chunk_size + 1; 
    end_idx = min((w+1) * chunk_size, n_total); 
    fprintf('\n the %d sliding window(from %d to %d, sum: %d) \n', ...
            w, start_idx, end_idx, end_idx - start_idx + 1);
    
    % --- get data from current wimdow ---
    % n x d
    X_win = X(start_idx:end_idx, :);
    L_win = X_label(start_idx:end_idx, :);
    % d x n
%     X_win = X(:, start_idx:end_idx)'; 
%     L_win = X_label(:, start_idx:end_idx)';

    % PCA 
    % [coeff, score, ~] = pca(X_win, 'NumComponents', 500);
    % X_win = score; 
    
    try
        [result, label_win, t_time, ddd_iter] = ERAPR(X_win, L_win, k, mu, eta, lambda);
%         [result, label_win, t_time, ddd_iter] = ablationR(X_win, L_win, k, mu, eta, lambda);

        all_results(w, :) = result;
        final_labels(start_idx:end_idx) = label_win;

        fprintf('window %d: ACC = %.4f, NMI = %.4f, Purity = %.4f, time = %.4fs\n', ...
        w, result(1), result(2), result(3), t_time);
    
    catch ME
        fprintf('window %d error: %s\n', w, ME.message);
    end
end

%% print results

valid_results = all_results(all_results(:,1) > 0, :);
avg_result = mean(all_results, 1);
fprintf('\n================result for this seed================\n');
fprintf('Avg ACC:    %.4f\n', avg_result(1));
fprintf('Avg NMI:    %.4f\n', avg_result(2));
% fprintf('Avg Purity: %.4f\n', avg_result(3));
% fprintf('Avg F1:     %.4f\n', avg_result(4));
% fprintf('Avg_win_time:  %.4fs\n', mean(all_results(:,5)));
% fprintf('total time:   %.4fs\n', sum(all_results(:,5)));
fprintf('========================================\n');

