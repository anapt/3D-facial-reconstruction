  function image = patch_and_show( position, color, cells )
% PATCH AND SAVE - Function that performs triangle filling and
%                  shading using patch and saves resulting plot
%                  using export_fig
%
% SYNTAX
%
%   patch_and_save
%
% INPUT
%
%   n           sequence number                         [  1  ]
%
% OUTPUT
%
%   void
%
% DESCRIPTION
%
%   patch_and_save implements a filling algorithm
%   and save the resulting plot, it reads the coordinates,
%   and color from txt files
%       position_n.txt
%       color_n.txt
%       cells_n.txt
%
% DEPENDENCIES
%
%   patch
%   export_fig
%

  %% PARAMETERS
  addpath('./altmany-export_fig-b1a7288')
  N = 53149;
  n_cells = 105694;
  basepath = './DATASET/';

  a = [position{:}];
  x = cell2mat(a);
  position = double(reshape(x, N, 2)).';

  a = [color{:}];
  x = cell2mat(a);
  color = double(reshape(x, N, 3)).';

  a = [cells{:}];
  x = cell2mat(a);
  cells = double(reshape(x, n_cells, 3)).';

  %% PATCH
  X = zeros(3, n_cells);
  Y = zeros(3, n_cells);
  C = ones(3, n_cells, 3);

  for i=1:1:n_cells
      X(:, i) = position(1, cells(:,i)+1);
      Y(:, i) = position(2, cells(:,i)+1);
      C(:, i, :) = color(:, cells(:,i)+1).';
  end

  %% EXPORT FIG
  close all
  figure('Position', [-1 -1 350 350])
  patch(X, Y, C, 'edgecolor','none');
  axis('tight', 'equal')
  axis off

  F = getframe(gcf);
  close all;
  [image, Map] = frame2im(F);

%  filename = sprintf('image_%d', 5);
%  f = [basepath 'matlab' filesep filename '.png'];
%  export_fig(f);
end

%%------------------------------------------------------------
%
% AUTHORS
%
%   Anastasia Pachni Tsitiridou          aipachni@ece.auth.gr
%
% VERSION
%
%   0.1 - June 27, 2019
%
% CHANGELOG
%
%   0.1 (Jun 27, 2019) - Anastasia