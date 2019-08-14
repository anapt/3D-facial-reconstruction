fileID = fopen('./average_face_2d_coords.txt','r');
shape = fscanf(fileID,"%f",2*53149);
shape = reshape(shape, [2, 53149]);

