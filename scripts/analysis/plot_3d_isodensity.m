% plot_3d_isodensity.m
% Render 3D isodensity surfaces with lighting.
%
% Usage:
%   Place this script in the directory containing fields_*.mat and run; for
%   another file, edit the file name in the load line below.
%
% Input:  fields_*.mat (3D L-FTS output, or CL-FTS output, whose real part is
%         drawn) or a 3D SCFT result .mat
% Output: isodensity.png
%
% The grid points sit at the cell centres, (i + 1/2)/n along each lattice
% vector, as in the solvers; the fields of a hexagonal-axes space group (SCFT
% results with space_group) sample the lattice axes a and b at the nodes, i/n.
% A non-orthogonal cell is drawn along its lattice vectors (a along x, b in
% the xy plane, as in ComputationBox).

clear all;

% Load data
load("fields_002000.mat");

% Get phi_A (handle different variable naming conventions)
if exist('phi_A','var')
    phi_a = phi_A;
elseif exist('phi_A_real','var')
    phi_a = phi_A_real;     % CL-FTS: the real part
elseif exist('phi','var')
    phi_a = phi.A;
end

% Reshape and permute for correct orientation
v = reshape(phi_a, [nx(3), nx(2), nx(1)]);
v = permute(v, [2 3 1]);
nx = double(nx);

% Lattice angles [alpha, beta, gamma] in degrees: saved by SCFT, in the
% checkpoint settings of L-FTS and CL-FTS fields files, or in the input of
% older files
if ~exist('angles','var')
    if exist('checkpoint_settings_json','var')
        settings = jsondecode(checkpoint_settings_json);
        angles = settings.angles(:)';
    elseif exist('initial_params','var') && isfield(initial_params, 'angles') && ~isempty(initial_params.angles)
        angles = double(initial_params.angles(:)');
    else
        angles = [90 90 90];
        fprintf("The file holds no lattice angles: the cell is drawn orthogonal.\n");
    end
end
rad = deg2rad(double(angles));

% Lattice vectors as rows
cx = cos(rad(2));
cy = (cos(rad(1)) - cos(rad(2))*cos(rad(3)))/sin(rad(3));
L = [lx(1)*[1 0 0];
     lx(2)*[cos(rad(3)) sin(rad(3)) 0];
     lx(3)*[cx cy sqrt(1 - cx^2 - cy^2)]];

% Fractional coordinates of the grid points: the cell centres, or the nodes
% along a and b for a hexagonal-axes space group (Hall numbers 430-488 except
% the rhombohedral-axes settings)
offsets = [0.5 0.5 0.5];
if exist('space_group_hall_number','var')
    hall = double(space_group_hall_number);
    if hall >= 430 && hall <= 488 && ~ismember(hall, [434 437 445 451 453 459 461])
        offsets(1:2) = 0;
    end
end
[fa, fb, fc] = meshgrid(((0:nx(1)-1) + offsets(1))/nx(1), ...
                        ((0:nx(2)-1) + offsets(2))/nx(2), ...
                        ((0:nx(3)-1) + offsets(3))/nx(3));

% Create figure
h = figure;

% Colormap
c = jet(1024);
colormap(c);
caxis([0.0 1.0]);

% Isovalue at mean density
isovalue = mean(phi_a);

% Draw isosurface: built in fractional coordinates and mapped into the cell,
% where a point is f*L and the gradient (the vertex normal) is n/L'
fv = isosurface(fa, fb, fc, v, isovalue);
normals = isonormals(fa, fb, fc, v, fv.vertices);
fv.vertices = fv.vertices * L;
p1 = patch(fv, 'VertexNormals', normals / L', 'FaceColor', c(1,:), 'EdgeColor', 'none');

% Draw isocaps
caps = isocaps(fa, fb, fc, v, isovalue);
caps.vertices = caps.vertices * L;
p2 = patch(caps, 'FaceColor', 'interp', 'EdgeColor', 'none');
alpha(p1, 0.5);

% View and lighting: the box around the cell
corners = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; 1 1 1] * L;
axis([min(corners(:,1)) max(corners(:,1)) min(corners(:,2)) max(corners(:,2)) ...
      min(corners(:,3)) max(corners(:,3))]);
lighting gouraud;
daspect([1 1 1]);
view(30, 20);
camlight right;

% Save figure
set(h, 'PaperPositionMode', 'auto');
set(h, 'PaperUnits', 'points');
set(h, 'PaperPosition', [0 0 800 500]);
print(h, 'isodensity', '-dpng');

fprintf("Saved to isodensity.png\n");
