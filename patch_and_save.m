function patch_and_save( n )
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
%   cells       triangles                               []
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
%
% DEPENDENCIES
%
%   patch
%   export_fig
%

  %% PARAMETERS
  N = 53149;
  n_cells = 105694;
  % dataset options
  basepath = './DATASET/';
  
  %% READ TXT FILES
  % COLOR
  filename = sprintf('color_%d', n);
  f = [basepath 'color' filesep filename '.txt'];
  fileID = fopen(f,'r');
  A = fscanf(fileID, '%f');
  
  color = zeros(3, N);
 
  color(1,:) = A(1:N);
  color(2,:) = A(N+1:(2*N));
  color(3,:) = A((2*N)+1:(3*N));
  
  % POSITION
  filename = sprintf('position_%d', n);
  f = [basepath 'position' filesep filename '.txt'];
  fileID = fopen(f,'r');
  A = fscanf(fileID, '%f');
  
  position = zeros(2, N);
 
  position(1,:) = A(1:N);
  position(2,:) = A(N+1:2*N);
  
  % CELLS
  filename = sprintf('cells_%d', n);
  f = [basepath 'cells' filesep filename '.txt'];
  fileID = fopen(f,'r');
  A = fscanf(fileID, '%f');
  
  cells = zeros(3, n_cells);
  
  cells(1,:) = A(1:n_cells);
  cells(2,:) = A(n_cells+1:(2*n_cells));
  cells(3,:) = A((2*n_cells)+1:(3*n_cells));
  
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
  
  filename = sprintf('im_%d', n);
  f = [basepath 'images' filesep filename '.png'];
  export_fig(f);
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