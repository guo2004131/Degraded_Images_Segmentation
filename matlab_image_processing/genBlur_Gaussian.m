clear; close all; clc;
%% setup 
% blur degree -- blur_delta
blur_delta = 3;
% dataset -- 'VOC', 'CamVid', 'SUNRGBD'
dataset = 'VOC';

%% do not change
out_pth = sprintf('/home/dg/Dropbox/Datasets/%s/Degraded_Images/Blur_Gaussian', dataset);

train_images_pth = sprintf('/home/dg/Dropbox/Datasets/%s/Original_Images/%s_train_images', dataset, dataset);
test_images_pth = sprintf('/home/dg/Dropbox/Datasets/%s/Original_Images/%s_test_images', dataset, dataset);

train_out_folder = fullfile(out_pth, sprintf('degraded_parameter_%d/%s_train_images', blur_delta, dataset));
test_out_folder = fullfile(out_pth, sprintf('degraded_parameter_%d/%s_test_images', blur_delta, dataset));

if ~exist(train_out_folder, 'dir')
    mkdir(train_out_folder);
end
if ~exist(test_out_folder, 'dir')
    mkdir(test_out_folder);
end
% images files
% images files
train_images = dir(fullfile(train_images_pth, '*.jpg'));
if size(train_images,1) == 0
    train_images = dir(fullfile(train_images_pth, '*.png'));
end
test_images = dir(fullfile(test_images_pth, '*.jpg'));
if size(test_images, 1) == 0
    test_images = dir(fullfile(test_images_pth, '*.png'));
end

% Game on
parfor i = 1:size(train_images, 1)
   fprintf('Now: %d/%d\n', i, size(train_images,1));
   I = imread(fullfile(train_images_pth, train_images(i).name));
   Iblur = imgaussfilt(I, blur_delta);
   output_filename = fullfile(train_out_folder, train_images(i).name);
   imwrite(Iblur, output_filename);
end

parfor i = 1:size(test_images, 1)
   fprintf('Now: %d/%d\n', i, size(test_images,1));
   I = imread(fullfile(test_images_pth, test_images(i).name));
   Iblur = imgaussfilt(I, blur_delta);
   output_filename = fullfile(test_out_folder, test_images(i).name);
   imwrite(Iblur, output_filename);
end