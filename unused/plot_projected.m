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
axis('equal')
scatter(projected(1,:),projected(2,:),5, colors.')

fileID = fopen('reflectance.txt','r');
A = fscanf(fileID, '%f');
reflectance = zeros(3, 53149);
reflectance(1,:) = A(1:53149);
reflectance(2,:) = A(53150:53149+53149);
reflectance(3,:) = A(53150+53149:159447);

figure(2)
axis('equal')
scatter(projected(1,:),projected(2,:),5, reflectance.')

norm(colors - reflectance)

fileID = fopen('normals.txt','r');
A = fscanf(fileID, '%f');
normals(1,:) = A(1:53149);
normals(2,:) = A(53150:53149+53149);
normals(3,:) = A(53150+53149:159447);

fileID = fopen('cells.txt','r');
A = fscanf(fileID, '%f');
cells(1,:) = A(1:105694);
cells(2,:) = A(105695:211388);
cells(3,:) = A(211389:317082);

X = zeros(3,105694);
Y = zeros(3,105694);
C = ones(3,105694, 3);

for i=1:1:105693
    X(:, i) = projected(1, cells(:,i)+1);
    Y(:, i) = projected(2, cells(:,i)+1);
    C(:, i, :) = colors(:, cells(:,i)+1).';
end

close all
figure('Position', [-1 -1 350 350])
bb = patch(X, Y, C, 'edgecolor','none')
axis('tight', 'equal')

axis off
export_fig('yourfigure.png');





% poly2mask(X,Y,C)
% marg = 0;
% rect = [-marg, -marg, 240, 240];
F = getframe(gca);
% figure(4)
image = F.cdata;
% imshow(image)