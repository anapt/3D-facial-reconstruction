clear
close all

%% PARAMETERS
N = 2500;
n_cells = 105694;
% dataset options

%% call patch and save

for i=0:1:(N-1)
    patch_and_save(i);
end