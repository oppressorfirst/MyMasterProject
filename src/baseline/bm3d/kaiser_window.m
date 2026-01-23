%% Kaiser 2D Window Visualization (MATLAB)
% Author: (your name)
% Usage: just run this script.

clear; clc; close all;

% ====== Parameters ======
n1   = 16;     % window size (e.g., 7, 11, 31)
beta = 1.0;    % Kaiser beta (e.g., 2~8)

% ====== Build 2D Kaiser Window ======
k1 = kaiser(n1, beta);         % 1D Kaiser
win2d = k1 * k1.';             % outer product -> 2D
win2d = single(win2d);         % float32 like numpy

% ====== Print info ======
fprintf("win2d: size = %dx%d, class = %s\n", size(win2d,1), size(win2d,2), class(win2d));
fprintf("min = %.6f, max = %.6f, mean = %.6f\n", min(win2d(:)), max(win2d(:)), mean(win2d(:)));

% If you want to print the whole matrix (only recommended for small n1):
if n1 <= 15
    disp("Full win2d matrix:");
    disp(win2d);
else
    disp("Top-left 6x6 of win2d:");
    disp(win2d(1:6, 1:6));
end

% ====== 1) Heatmap (imagesc) ======
figure('Name','2D Kaiser Window - Heatmap');
imagesc(win2d);
axis image;
colorbar;
title(sprintf('2D Kaiser Window (n1=%d, \\beta=%.2f)', n1, beta));
xlabel('x'); ylabel('y');

% ====== 2) 3D Surface ======
figure('Name','2D Kaiser Window - Surface');
[X, Y] = meshgrid(1:n1, 1:n1);
surf(X, Y, double(win2d), 'EdgeColor', 'none'); % smooth surface
colorbar;
title(sprintf('3D Surface of 2D Kaiser (n1=%d, \\beta=%.2f)', n1, beta));
xlabel('x'); ylabel('y'); zlabel('weight');
view(45, 35);
grid on;

% ====== 3) Center cross-section curves ======
mid = floor(n1/2) + 1;
center_row = win2d(mid, :);
center_col = win2d(:, mid);

figure('Name','2D Kaiser Window - Center Cross-section');
plot(1:n1, center_row, 'LineWidth', 2); hold on;
plot(1:n1, center_col, 'LineWidth', 2);
grid on;
legend('center row', 'center col', 'Location', 'best');
title(sprintf('Center Cross-section (n1=%d, \\beta=%.2f)', n1, beta));
xlabel('index'); ylabel('weight');

% ====== Optional: save figures ======
% exportgraphics(gcf, 'kaiser_cross.png', 'Resolution', 200);
% saveas(findobj('Name','2D Kaiser Window - Heatmap'), 'kaiser_heatmap.png');
% saveas(findobj('Name','2D Kaiser Window - Surface'), 'kaiser_surface.png');
