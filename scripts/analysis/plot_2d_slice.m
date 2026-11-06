% plot_2d_slice.m
% Extract and plot a 2D slice from 3D density data.
%
% Usage:
%   Place this script in the directory containing fields_*.mat and run; for
%   another file, edit the file name in the load line below.
%
% Input:  fields_*.mat (3D L-FTS output, or CL-FTS output, whose real part is
%         drawn) or a 3D SCFT result .mat
% Output: 2d_slice_image.png
%
% The slice is the plane of grid points with index slice_index along a,
% spanned by b and c. Each value fills the grid cell around its point, which
% sits at (j + 1/2)/n along each lattice vector as in the solvers (at j/n
% along b for a hexagonal-axes space group), so the image covers the plane
% with the lengths of b and c and the angle alpha between them.

clear all;

% Load data
load("fields_002000.mat");
slice_index = 1;        % 1 to nx(1)

% Get phi_A (handle different variable naming conventions)
if exist('phi_A','var')
    phi_a = phi_A;
elseif exist('phi_A_real','var')
    phi_a = phi_A_real;     % CL-FTS: the real part
elseif exist('phi','var')
    phi_a = phi.A;
end

% Display min/max values
fprintf("phi_A min: %.4f, max: %.4f\n", min(phi_a(:)), max(phi_a(:)));

% Reshape to the 3D grid, data(k, j, i), and take the plane of index i:
% rows along c, columns along b
data = reshape(phi_a, [nx(3), nx(2), nx(1)]);
phi_a_2d = data(:, :, slice_index);
nx = double(nx);

% The angle alpha between b and c in degrees: SCFT results, the checkpoint
% settings of fields files, or the input of older files
if exist('angles','var')
    alpha_deg = double(angles(1));
elseif exist('checkpoint_settings_json','var')
    settings = jsondecode(checkpoint_settings_json);
    alpha_deg = settings.angles(1);
elseif exist('initial_params','var') && isfield(initial_params, 'angles') && ~isempty(initial_params.angles)
    alpha_deg = double(initial_params.angles(1));
else
    alpha_deg = 90;
    fprintf("The file holds no lattice angles: the plane is drawn rectangular.\n");
end
b = lx(2)*[1 0];
c = lx(3)*[cosd(alpha_deg) sind(alpha_deg)];

% A hexagonal-axes space group (Hall numbers 430-488 except the
% rhombohedral-axes settings) samples b at the nodes
offset_b = 0.5;
if exist('space_group_hall_number','var')
    hall = double(space_group_hall_number);
    if hall >= 430 && hall <= 488 && ~ismember(hall, [434 437 445 451 453 459 461])
        offset_b = 0;
    end
end

% Corners of the grid cells; pcolor colors each cell by the value at its first
% corner, so the last row and column of the colors (copies, which keep the
% automatic color limits) are not drawn
[fb, fc] = meshgrid(((0:nx(2)) + offset_b - 0.5)/nx(2), (0:nx(3))/nx(3));
C = phi_a_2d([1:end end], [1:end end]);

% Plot
h = figure;
pcolor(fb*b(1) + fc*c(1), fb*b(2) + fc*c(2), C);
shading flat;
axis equal;
axis off;

% Colormap
colormap(jet(1024));
caxis([0.0 1.0]);
colorbar('FontSize', 20);

% Save figure
set(h, 'PaperPositionMode', 'auto');
set(h, 'PaperUnits', 'points');
set(h, 'PaperPosition', [0 0 500 500]);
print(h, "2d_slice_image", '-dpng');

fprintf("Saved to 2d_slice_image.png\n");
