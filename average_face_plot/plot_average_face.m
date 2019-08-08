fileID = fopen('./avg_shape.txt','r');
shape = fscanf(fileID,"%f",3*53149);
shape = reshape(shape, [3, 53149]);

fileID = fopen('./avg_reflectance.txt','r');
A = fscanf(fileID,"%f",3*53149);
reflectance = reshape(A, [3,53149]);

figure(1)
hold on
axis equal
scatter3(shape(1,:),shape(2,:),shape(3,:),1, reflectance.');