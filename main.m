close all
clear;

addpath('ompbox10');

set(0, 'DefaultAxesFontSize',  10);
set(0, 'DefaultLineLineWidth', 3);

% I = imread('forest.tif');
% I = imread('blobs.png');
% I = imread('canoe.tif');
% I = imread('football.jpg');
I = imread('barbara.png');

I = mean(double(I), 3);
I = I(301:500, 301:500);
figure; imshow(I, []);

run_part_A = false;
run_part_B = false;

% run_part_A = true;
run_part_B = true;

%%
patch_width    = 8;
patch_height   = 8;
patch_size     = [patch_height, patch_width];
atom_length    = patch_width * patch_height;
cardinality    = 3;
signature_size = [30, 30];
training_set_size = 5000;
%%

if run_part_A == true
%-- Create Super Set from the Image:
mSuper_set = im2col(I, patch_size);

%-- Remove mean:
mSuper_set = bsxfun(@minus, mSuper_set, mean(mSuper_set, 1));

vTrain_set_idx    = randperm(length(mSuper_set), training_set_size);
mTrain            = mSuper_set(:, vTrain_set_idx);

%% Batch:
if 1
[mSD_batch, mSD0_batch, vF_coef, vF_dict]   = Signature_Dictionary_Learninig_Batch(...
                      signature_size, mTrain, patch_size, cardinality);

%% Pursuit:
mD = im2col(mSD_batch, [patch_height, patch_width]);
vW = sqrt( sum(mD.^2, 1) );
mA = bsxfun(@rdivide, mD, vW);
mG = mA' * mA;
mX = omp(mA' * mSuper_set, mG, cardinality);

mR         = mA * mX - mSuper_set;
RMSE_batch = sqrt( mean( mean((mR).^2, 1) ) );

mX              = bsxfun(@rdivide, mX, vW');
vActivity_batch = sum(abs(mX), 2);
%%

end

%% On Line:
[mSD_on_line, mSD0_on_line, vF_on_line]   = Signature_Dictionary_Learninig_On_Line(...
                      signature_size, mTrain, patch_size, cardinality);

%% Pursuit:
mD = im2col(mSD_on_line, [patch_height, patch_width]);
vW = sqrt( sum(mD.^2, 1) );
mA = bsxfun(@rdivide, mD, vW);
mG = mA' * mA;
mX = omp(mA' * mSuper_set, mG, cardinality);

mR           = mA * mX - mSuper_set;
RMSE_on_line = sqrt( mean( mean((mR).^2, 1) ) );

mX                = bsxfun(@rdivide, mX, vW');
vActivity_on_line = sum(abs(mX), 2);

%%
figure; hold on;
plot(vF_coef,    'b');
plot(vF_dict,    ':g');
plot(vF_on_line, 'r');
legend('After Coef Update', ...
       'After Dict Update', ...
       'On Line', 0);

mActivity_batch   = reshape(vActivity_batch,   signature_size - patch_size + 1);
mActivity_on_line = reshape(vActivity_on_line, signature_size - patch_size + 1);

figure;
subplot(2,3,1); imshow(mSD0_batch,   []); title('Batch Init');  
subplot(2,3,2); imshow(mSD_batch,    []); title(['Batch Final, RMSE = ', num2str(RMSE_batch)]); 
subplot(2,3,4); imshow(mSD0_on_line, []); title('On Line Init'); 
subplot(2,3,5); imshow(mSD_on_line,  []); title(['On Line Final, RMSE = ', num2str(RMSE_on_line)]);
subplot(2,3,3); imagesc(mActivity_batch);   colorbar; title('Bacth Activity');
subplot(2,3,6); imagesc(mActivity_on_line); colorbar; title('On Line Activity');
end

%%
if run_part_B == true
std     = 20;
I_noisy = I + std * randn(size(I));

%-- Create Super Set from the Image:
mSuper_set = im2col(I, patch_size);
%-- Remove mean:
vSuper_set_mean = mean(mSuper_set, 1);
mSuper_set      = bsxfun(@minus, mSuper_set, vSuper_set_mean);

vTrain_set_idx    = randperm(length(mSuper_set), training_set_size);
mTrain            = mSuper_set(:, vTrain_set_idx);

thrshold = 1.15 * prod(patch_size) * std^2;
mSD  = Signature_Dictionary_Learninig_On_Line2(...
                signature_size, mTrain, patch_size, thrshold);

end