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
% I = imread('0002.png');

I = mean(double(I), 3);
% I = I(301:500, 301:500);
% I = I(1:300, 1:300);
% figure; imshow(I, []);

run_part_A = false;
run_part_B = false;

% run_part_A = true;
run_part_B = true;

%%
patch_width       = 8;
patch_height      = 8;
vPatch_size       = [patch_height, patch_width];
atom_length       = patch_width * patch_height;
cardinality       = 3;
vSignature_size   = [30, 30];
training_set_size = 5000;

%-- PSNR Function
RefImageDR = @(vY0)     max(vY0) - min(vY0);
CalcPsnr   = @(vY, vY0) pow2db( (length(vY) * RefImageDR(vY0) ^ 2) / (norm(vY - vY0) ^ 2) );

%%
if run_part_A == true
%-- Create Super Set from the Image:
mSuper_set = im2col(I, vPatch_size);

%-- Remove mean:
vSuper_set_mean = mean(mSuper_set, 1);
mSuper_set      = bsxfun(@minus, mSuper_set, vSuper_set_mean);

vTrain_set_idx  = randperm(length(mSuper_set), training_set_size);
mTrain          = mSuper_set(:, vTrain_set_idx);

%% Batch:
if 1
[mSD_batch, mSD0_batch, vF_coef, vF_dict] = ...
                              Signature_Dictionary_Learninig_Batch(...
                                vSignature_size, mTrain, vPatch_size, cardinality);

%% Pursuit:
mD = im2col(mSD_batch, [patch_height, patch_width]);
vW = sqrt( sum(mD.^2, 1) );
mA = bsxfun(@rdivide, mD, vW);
mG = mA' * mA;
mX = omp(mA' * mSuper_set, mG, cardinality);

mR         = mA * mX - mSuper_set;
RMSE_batch = sqrt( mean( mean((mR).^2, 1) ) );

mA_batch        = mA;
mX              = bsxfun(@rdivide, mX, vW');
vActivity_batch = sum(abs(mX), 2);

%%
mP     = bsxfun(@plus, mD * mX, vSuper_set_mean);
mBatch = Col_To_Im(mP, size(I), vPatch_size);

end

%% On Line:
[mSD_on_line, mSD0_on_line, vF_on_line]   = Signature_Dictionary_Learninig_On_Line(...
                      vSignature_size, mTrain, vPatch_size, cardinality);

%% Pursuit:
mD = im2col(mSD_on_line, [patch_height, patch_width]);
vW = sqrt( sum(mD.^2, 1) );
mA = bsxfun(@rdivide, mD, vW);
mG = mA' * mA;
mX = omp(mA' * mSuper_set, mG, cardinality);

mR           = mA * mX - mSuper_set;
RMSE_on_line = sqrt( mean( mean((mR).^2, 1) ) );

mA_on_line        = mA;
mX                = bsxfun(@rdivide, mX, vW');
vActivity_on_line = sum(abs(mX), 2);

%%
mP       = bsxfun(@plus, mD * mX, vSuper_set_mean);
mOn_line = Col_To_Im(mP, size(I), vPatch_size);

%%
figure; hold on;
plot(vF_coef,    'b');
plot(vF_dict,    ':g');
plot(vF_on_line, 'r');
legend('After Coef Update', ...
       'After Dict Update', ...
       'On Line', 0);

mActivity_batch   = reshape(vActivity_batch,   vSignature_size - vPatch_size + 1);
mActivity_on_line = reshape(vActivity_on_line, vSignature_size - vPatch_size + 1);

figure;
Y = 2;
X = 4;
subplot(Y,X,1); imshow(mSD0_batch,   []); title('Batch Init');  
subplot(Y,X,5); imshow(mSD0_on_line, []); title('On Line Init'); 
subplot(Y,X,2); imshow(mSD_batch,    []); title('Batch Final'); 
subplot(Y,X,6); imshow(mSD_on_line,  []); title('On Line Final');
subplot(Y,X,3); imagesc(mActivity_batch);   colorbar; title('Bacth Activity');
subplot(Y,X,7); imagesc(mActivity_on_line); colorbar; title('On Line Activity');
subplot(Y,X,4); imshow(mBatch,       []); title(['Batch Rec, RMSE = ',   num2str(RMSE_batch)]); 
subplot(Y,X,8); imshow(mOn_line,     []); title(['On Line Rec, RMSE = ', num2str(RMSE_on_line)]);

figure;
subplot(1,2,1); Display_D(mA_batch);
subplot(1,2,2); Display_D(mA_on_line);


end

%%
if run_part_B == true
std     = 20;
I_noisy = I + std * randn(size(I));
% I_noisy = I;

%-- Create Super Set from the Image:
mSuper_set = im2col(I_noisy, vPatch_size);
%-- Remove mean:
vSuper_set_mean = mean(mSuper_set, 1);
mSuper_set      = bsxfun(@minus, mSuper_set, vSuper_set_mean);

vTrain_set_idx    = randperm(length(mSuper_set), training_set_size);
mTrain            = mSuper_set(:, vTrain_set_idx);
tic;
threshold = sqrt(1.15 * prod(vPatch_size) * std^2);
[mSD, vMean_suppoert, vF]  = Signature_Dictionary_Learninig_On_Line2(...
                              vSignature_size, mTrain, vPatch_size, threshold);
toc;

tic;
mD   = im2col(mSD, [patch_height, patch_width]);
vW   = sqrt( sum(mD.^2, 1) );
mA   = bsxfun(@rdivide, mD, vW);
mG   = mA' * mA;
mX   = omp2(mA' * mSuper_set, sum(mSuper_set .* mSuper_set), mG, threshold);      
% mX    = omp2(mA, mSuper_set, mG, threshold);
mX    = bsxfun(@rdivide, mX, vW');
mP    = bsxfun(@plus, mD * mX, vSuper_set_mean);
toc();
mRec  = Col_To_Im(mP, size(I), vPatch_size);           
mMean = Col_To_Im(ones(atom_length, 1) * vSuper_set_mean, size(I), vPatch_size);           

figure; imshow(mSD, []);

figure;
x = 1 : length(vF);
[hAx, hLine1, hLine2] = plotyy(x, vMean_suppoert, x, vF);

title('On Line With Noisy Inputs');
xlabel('Iteration');

ylabel(hAx(1), 'Mean Cardinality');
ylabel(hAx(2), 'RMSE');

noise_psnr    = CalcPsnr(I_noisy(:), I(:));
rec_psnr      = CalcPsnr(mRec(:), I(:));
rec_mean_psnr = CalcPsnr(mMean(:), I(:));

I_noisy = max(min(I_noisy, 255), 0);
figure;
subplot(2,2,1); imshow(I,       []); title('Original');
subplot(2,2,2); imshow(I_noisy, []); title(['Noisy, PSNR = ', num2str(noise_psnr)]);
subplot(2,2,3); imshow(mRec,    []); title(['Reconstructed, PSNR = ', num2str(rec_psnr)]);
subplot(2,2,4); imshow(mMean,   []); title(['Mean Only, PSNR = ', num2str(rec_mean_psnr)]);

figure;
imshow([I, mRec], []);
end
