clear;
clc;
addpath PCA_result;
addpath EntropyRateSuperpixel-master;

result_path = fullfile('superpixel_result2');
%% main
mkdir(result_path);

% dataname = ["Salinas_corrected"; "Indian_pines_corrected"; "PaviaU"; "Houston"];
dataname = ["WHU_Hi_LongKou"];
% superpixel_num = linspace(1000, 2000, 11);
superpixel_num = linspace(2100, 2500, 5);
disp(superpixel_num);
for i=1:1
    idataname = dataname(i, :);
    filename = strcat('./PCA_result/', idataname , '_pca.mat');
    load(filename);
    for i_sp_num = 1:length(superpixel_num)
        sp_map = Superpixel_func(double(data), superpixel_num(i_sp_num));
        sp_name = strcat(idataname , '_sp_num_', num2str(superpixel_num(i_sp_num)), '.mat');
        sp_path = fullfile(result_path, sp_name);
        save(sp_path, 'sp_map')
    end
    disp(filename)
end