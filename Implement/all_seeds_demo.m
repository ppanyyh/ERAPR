clear; 

%% data input

DataName = 'nci9';
load([DataName,'_data.mat'])

X_orig = double(X);
X_orig = X_orig ./ (max(X_orig(:)) + 1e-9);
X_label_orig = Y;

%% parameter settings 

seeds = [3, 123, 1337, 2026, 1, 1024, 777, 31415, 99, 265]; 
num_experiments = length(seeds);

chunk_size = 10;      % chunk size for window
k = 2; 
lambda = 100;
mu = 1;
eta = 1;

%% 10 independent randomized experiments

% results of 10 seeds 
summary_table = zeros(num_experiments, 6); 
fprintf('dataset: %s, 10 experiments \n', DataName);

for s = 1:num_experiments
    current_seed = seeds(s);
    fprintf('\n the %d/10 experiment | seed: %d \n', s, current_seed);
    
    % shuffled data
    rng(current_seed);
    p = randperm(size(X_orig, 1));
    X = X_orig(p, :);
    X_label = X_label_orig(p, :);
    
    % define window
    [n_total, dim] = size(X);
    num_chunks = ceil(n_total / chunk_size);
    num_sliding_windows = num_chunks - 1;
    win_results = zeros(num_sliding_windows, 5);  % results of all windows of the current seed

    for w = 1:num_sliding_windows
        start_idx = (w-1) * chunk_size + 1;
        end_idx = min((w+1) * chunk_size, n_total);
        
        % get data from current wimdow 
        % n x d
        X_win = X(start_idx:end_idx, :);
        L_win = X_label(start_idx:end_idx, :);
        % d x n
%         X_win = X(:, start_idx:end_idx)';
%         L_win = X_label(:, start_idx:end_idx)';

        % PCA 
        % [coeff, score, ~] = pca(X_win, 'NumComponents', 500);
        % X_win = score; 
        
        try
            [result, label_win, t_time, ddd_iter] = ERAPR(X_win, L_win, k, mu, eta, lambda);
            % [result, label_win, t_time, ddd_iter] = ablationQ(X_win, L_win, k, mu, eta, lambda);
            win_results(w, :) = result;  % include [ACC, NMI, Purity, F1, Time]
        
        catch ME
            fprintf('window %d error: %s\n', w, ME.message);
        end
    end

    % avg results of the current seed
    valid_idx = win_results(:, 1) > 0;
    valid_win_res = win_results(valid_idx, :);
    seed_acc = mean(valid_win_res(:, 1));
    seed_nmi = mean(valid_win_res(:, 2));
    seed_pur = mean(valid_win_res(:, 3));
    seed_f1  = mean(valid_win_res(:, 4));
    seed_avg_time = mean(valid_win_res(:, 5));
    seed_total_time = sum(valid_win_res(:, 5));
    summary_table(s, :) = [seed_acc, seed_nmi, seed_pur, seed_f1, seed_avg_time, seed_total_time];
    fprintf('seed %d : ACC=%.4f, NMI=%.4f, total time=%.2fs\n', ...
            current_seed, seed_acc, seed_nmi, seed_total_time);
end

%% final results of 10 experiments

final_mean = mean(summary_table, 1);
final_std = std(summary_table, 0, 1);
fprintf('%s\n', repmat('=', 1, 50)); 
fprintf('final results - dataset: %s\n', DataName);
fprintf('%d independent randomized experiments\n', num_experiments);

fprintf('%s\n', repmat('-', 1, 50)); 
fprintf('Avg ACC:    %.4f ± %.4f\n', final_mean(1), final_std(1));
fprintf('Avg NMI:    %.4f ± %.4f\n', final_mean(2), final_std(2));
% fprintf('Avg Purity: %.4f ± %.4f\n', final_mean(3), final_std(3));
% fprintf('Avg F1:     %.4f ± %.4f\n', final_mean(4), final_std(4));
% fprintf('Avg_Win_Time: %.4fs\n', final_mean(5));
% fprintf('Avg_seed_Time: %.4fs\n', final_mean(6));
fprintf('%s\n', repmat('=', 1, 50)); 

% output table
T = array2table(summary_table, 'VariableNames', ...
    {'ACC', 'NMI', 'Purity', 'F1', 'Avg_Win_Time', 'Avg_seed_Time'});
writetable(T, ['Results_', DataName, '.xlsx']);