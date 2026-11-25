%% Multi rng evaluation logic
function [mean_acc, mean_nmi] = run_multi_seed_eval(X_orig, L_orig, seeds, c_size, k, lambda, mu, eta)
    acc_list = [];
    nmi_list = [];
    
    for s = 1:length(seeds)
        rng(seeds(s));
        p = randperm(size(X_orig, 1));
        X = X_orig(p, :);
        L = L_orig(p, :);
        
        [n_total, dim] = size(X_orig);
        num_chunks = ceil(size(X, 1) / c_size);
        n_win = num_chunks - 1;
        curr_acc = zeros(n_win, 1);
        curr_nmi = zeros(n_win, 1);
        
        for w = 1:n_win
            idx = (w-1)*c_size+1 : min((w+1)*c_size, n_total);
            X_win = X(idx, :); 
            L_win = L(idx, :);
            
            % PCA only nci9
            % n_comp = min(size(X_win, 1)-1, 30)
            % [coeff, score, ~] = pca(X_win, 'NumComponents', n_comp);
            % X_win = score; 

            [res, ~, ~, ~] = ERAPR(X_win, L_win, k, mu, eta, lambda);
            curr_acc(w) = res(1);
            curr_nmi(w) = res(2);
        end
        acc_list = [acc_list; mean(curr_acc)];
        nmi_list = [nmi_list; mean(curr_nmi)];
    end
    mean_acc = mean(acc_list);
    mean_nmi = mean(nmi_list);
end