clear
close all

img = imread('./000000.png');
figure(1)
histogram(img(:,:,1))

figure(2)
histogram(img(:,:,2))

figure(3)
histogram(img(:,:,3))

img = imread('./image_000000.png');
figure(4)
histogram(img(:,:,1))
mean = mean(img(:,:,1));
figure(5)
histogram(img(:,:,2))

figure(6)
histogram(img(:,:,3))

img = imread('./000000.png');

img(:, :, 1) = img(:, :, 1) + 50;
img(:, :, 2) = img(:, :, 2) + 100;
img(:, :, 3) = img(:, :, 3) + 20;