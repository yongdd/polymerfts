% plot_2d_density.m
% Plot 2D monomer density from simulation output.
%
% Usage:
%   Place this script in the directory containing fields.mat and run; for
%   another file, edit the file name in the load line below.
%
% Input:  fields.mat (2D simulation output)
% Output: 2d_density.png
%
% Each value fills the grid cell around its point, (i + 1/2)/n along each
% lattice vector as in the solvers, so the image covers the cell with its
% lengths lx and angle gamma (a along x, b in the xy plane).

clear all;

% Load data
load("fields.mat");

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

% Reshape to 2D grid: rows along b, columns along a
phi_a_2d = reshape(phi_a, [nx(2), nx(1)]);
nx = double(nx);

% The angle gamma between a and b in degrees: the last of the saved angles
% (SCFT results, the checkpoint settings of fields files, or the input)
if exist('angles','var')
    gamma_deg = double(angles(end));
elseif exist('checkpoint_settings_json','var')
    settings = jsondecode(checkpoint_settings_json);
    gamma_deg = settings.angles(end);
elseif exist('initial_params','var') && isfield(initial_params, 'angles') && ~isempty(initial_params.angles)
    gamma_deg = double(initial_params.angles(end));
else
    gamma_deg = 90;
    fprintf("The file holds no lattice angles: the cell is drawn rectangular.\n");
end
a = lx(1)*[1 0];
b = lx(2)*[cosd(gamma_deg) sind(gamma_deg)];

% Corners of the grid cells; pcolor colors each cell by the value at its first
% corner, so the last row and column of the colors (copies, which keep the
% automatic color limits) are not drawn
[fa, fb] = meshgrid((0:nx(1))/nx(1), (0:nx(2))/nx(2));
C = phi_a_2d([1:end end], [1:end end]);

% Plot
h = figure;
pcolor(fa*a(1) + fb*b(1), fa*a(2) + fb*b(2), C);
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
print(h, "2d_density", '-dpng');

fprintf("Saved to 2d_density.png\n");
