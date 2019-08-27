hinfo = hdf5info('/home/anapt/PycharmProjects/thesis/DATASET/model2017-1_bfm_nomouth.h5');
shape_pca = hdf5read(hinfo.GroupHierarchy.Groups(5).Groups(1).Datasets(3));
shape_var = hdf5read(hinfo.GroupHierarchy.Groups(5).Groups(1).Datasets(4));

figure(1)
xlim([0 80])
bar(shape_var(1:64))
 
figure(2)
xlim([0 80])
bar(shape_var(2:64))

sum(shape_var(1:64))/sum(shape_var)

shape_var(64)/sum(shape_var)*100
expression_var = hdf5read(hinfo.GroupHierarchy.Groups(3).Groups(1).Datasets(4));

figure(3)
xlim([0 80])
bar(expression_var(1:64))
 

sum(expression_var(1:64))/sum(expression_var)

color_var = hdf5read(hinfo.GroupHierarchy.Groups(2).Groups(1).Datasets(4));

figure(5)
xlim([0 100])
bar(color_var(1:100))


sum(color_var(1:100))/sum(color_var)
