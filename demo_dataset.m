clear
close all

%% PARAMETERS
N = 100;
n_cells = 105694;
% dataset options
basepath = './DATASET/';
  
%% READ CELLS
% CELLS
filename = sprintf('cells');
f = [basepath filesep filename '.txt'];
fileID = fopen(f,'r');
A = fscanf(fileID, '%f');
  
cells = zeros(3, n_cells);
 
cells(1,:) = A(1:n_cells);
cells(2,:) = A(n_cells+1:(2*n_cells));
cells(3,:) = A((2*n_cells)+1:(3*n_cells));

%% call patch and save

for i=0:1:100
    patch_and_save(i, cells);
end