function K = gaussian_kernel_2d(sigma, radius)
%GAUSSIAN_KERNEL_2D 手写生成 2D Gaussian kernel（等价于你的 Python 版本）
%   sigma : 标准差
%   radius: 半径（默认 floor(3*sigma)）
%
% 返回:
%   K : (2*radius+1) x (2*radius+1) 的归一化核（sum(K(:))=1）

    if nargin < 2 || isempty(radius)
        radius = floor(3 * sigma);
    end

    sz = 2 * radius + 1;
    K = zeros(sz, sz, 'single');

    for i = 1:sz
        for j = 1:sz
            x = (i - 1) - radius;   % 对齐 Python: x = i - radius (i从0开始)
            y = (j - 1) - radius;
            K(i, j) = exp(-(x*x + y*y) / (2 * sigma * sigma));
        end
    end

    K = K / sum(K(:));  % 归一化
end
