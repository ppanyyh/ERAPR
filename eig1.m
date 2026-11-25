function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)

% 在ERCAN中，[F,~,~,ev] = eig1(L_s, c, 0);
% 这意味着：
% 对拉普拉斯矩阵 L_s 进行特征分解
% 对特征值进行升序排列
% 返回最小的 c 个特征值及其对应的特征向量，并赋值给 F

if nargin < 2
    c = size(A,1);
    isMax = 1; % 1 表示返回最大的 c 个，0 表示返回最小的 c 个
    isSym = 1; % 1 表示强制将输入矩阵 A 对称化
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A');
    % 特征值分解理论上要求矩阵是对称的（对于实对称矩阵，特征值必为实数，特征向量两两正交）。在数值计算中，由于精度问题，拉普拉斯矩阵可能不是严格对称的，这一步可以保证其对称性，增加计算的稳定性。
end;
[v d] = eig(A); % 调用 MATLAB 内置函数进行特征分解。v 是特征向量矩阵（每列一个向量），d 是对角阵，对角线元素是特征值。
d = diag(d); % 将特征值从对角矩阵中提取出来，变成一个列向量。
%d = real(d);
if isMax == 0
    [d1, idx] = sort(d); % 如果 isMax == 0 (寻找最小值)，则使用 sort(d) 对特征值进行升序排列。
else
    [d1, idx] = sort(d,'descend'); % 否则 (寻找最大值)，使用 sort(d,'descend') 进行降序排列。
end;

idx1 = idx(1:c); % 提取排序后前 c 个元素的原始索引
eigval = d(idx1); % 根据这些索引，从"原始（未排序）"的特征值向量 d 中挑出对应的 c 个特征值
eigvec = v(:,idx1); % 根据索引 idx1，从"原始（未排序）"的特征向量矩阵 v 中挑出对应的 c 列

eigval_full = d(idx); % 返回所有排序后的特征值，方便调试