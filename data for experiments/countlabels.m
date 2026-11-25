% 假设 Y 是你的标签向量
% load('coil100_data.mat');
% load('Coil20_data.mat');
% load('forest_cover_data.mat'); 
% load('warpAR10P_data.mat'); 
% load('Yale_data.mat'); 
% load('nci9_data.mat'); 
load('abalone.mat');
% load('key_data.mat');
% load('network_data.mat');
% load('USPS_data.mat');
% load('ORL_data.mat');

% Y = gnd(:);
% Y = label(:);
% Y = forest_cover_labels(:); 
Y = Y(:);
% Y = Y(:);
% Y = key_labels(:);
% Y = network_labels(:);
% Y = Y(:);
%  Y = label(:);


[categories, ~, idx] = unique(Y);
counts = accumarray(idx, 1);
percentages = (counts / sum(counts)) * 100; % 计算百分比

stats = tabulate(Y);
fprintf('%10s %15s %15s\n', '类别标签', '样本数量', '百分比(%)');
for i = 1:size(stats, 1)
    % %d 表示整数，%.f 表示不带小数点的浮点数，%.2f 表示保留两位小数
    fprintf('%10d %15.0f %15.2f%%\n', stats(i,1), stats(i,2), stats(i,3));
end