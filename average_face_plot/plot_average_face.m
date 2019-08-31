fileID = fopen('./avg_shape.txt','r');
shape = fscanf(fileID,"%f",3*53149);
shape = reshape(shape, [3, 53149]);

fileID = fopen('./avg_reflectance.txt','r');
A = fscanf(fileID,"%f",3*53149);
reflectance = reshape(A, [3,53149]);

fileID = fopen('./cells.txt','r');
A = fscanf(fileID,"%f",3*105694);
cells = reshape(A, [3,105694]);

figure(1)
hold on
axis equal
xlabel('x - axis')
ylabel('y - axis')
zlabel('z - axis')
scatter3(shape(1,:),shape(2,:),shape(3,:),1, reflectance.');

rotation(0,45,0)

shape2 = rotation(0, 90, 0) * shape;

figure(2)
hold on
axis equal
xlabel('x - axis')
ylabel('y - axis')
zlabel('z - axis')
scatter3(shape2(1,:),shape2(2,:),shape2(3,:),1, reflectance.');