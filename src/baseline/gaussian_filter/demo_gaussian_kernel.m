clear; clc; close all;

sigma = 0.1;
K = gaussian_kernel_2d(sigma,9);

% 2D 热力图
figure('Name','Gaussian kernel - heatmap');
imagesc(K);
axis image; axis xy;
colorbar;
title(sprintf('Gaussian kernel (sigma=%.3f), size=%dx%d', sigma, size(K,1), size(K,2)));

% 3D 曲面
figure('Name','Gaussian kernel - surface');
[X, Y] = meshgrid(1:size(K,2), 1:size(K,1));
surf(X, Y, K, 'EdgeColor','none');
colorbar;
title(sprintf('Gaussian kernel surface (sigma=%.3f)', sigma));
xlabel('j'); ylabel('i'); zlabel('K(i,j)');
view(45, 30);
grid on;
