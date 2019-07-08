clear
close all

% Shape PCA Basis 
data = hdf5read ...
    ('./DATASET/model2017-1_bfm_nomouth.h5','shape/model/pcaBasis');

data = data(1:80,:).';
sdeviation = std(data);
sdev = sdeviation;
variance = var(data);

% s = data.' * data;
data1 = data(1,:);
variance1 = var(data1);
sdeviation1 = std(data1);
s = sdeviation1^0.5;
data1 = data1*s;
s1 = (data1).' * (data1);
s1(1, 1)

data2 = data(2,:);
variance2 = var(data1);
sdeviation2 = std(data1);
s = sdeviation2^0.5;
data2 = data2*s;
s2 = (data2).' * (data2);
s2(2, 2)

s = (data .* sdev).' * (data .* sdev);
s(1,1)
s(2,2)
