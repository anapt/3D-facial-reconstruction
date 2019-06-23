clear
close all

fileID = fopen('color.txt','r');
A = fscanf(fileID, '%f');
colors = zeros(3, 53149);
 
colors(1,:) = A(1:53149);
colors(2,:) = A(53150:53149+53149);
colors(3,:) = A(53150+53149:159447);

fileID = fopen('position.txt','r');
A = fscanf(fileID, '%f');
projected = zeros(2, 53149);
 
projected(1,:) = A(1:53149);
projected(2,:) = A(53150:53149+53149);

figure(1)
scatter(projected(1,:),projected(2,:),5, colors.')

fileID = fopen('reflectance.txt','r');
A = fscanf(fileID, '%f');
reflectance = zeros(3, 53149);
reflectance(1,:) = A(1:53149);
reflectance(2,:) = A(53150:53149+53149);
reflectance(3,:) = A(53150+53149:159447);

figure(2)
scatter(projected(1,:),projected(2,:),5, reflectance.')

norm(colors - reflectance)

fileID = fopen('normals.txt','r');
A = fscanf(fileID, '%f');
normals(1,:) = A(1:53149);
normals(2,:) = A(53150:53149+53149);
normals(3,:) = A(53150+53149:159447);

% figure(3)
% scatter3(normals(1,:), normals(2,:), normals(3,:), 1);