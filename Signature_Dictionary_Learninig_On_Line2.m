function [mSD, vMean_support, vF] = Signature_Dictionary_Learninig_On_Line2(...
                  vSignature_size, mTrain, vPatch_size, threshold)

vD_idx = 1 : prod(vSignature_size);
mD_idx = reshape(vD_idx, vSignature_size);
mD_idx = im2col(mD_idx, vPatch_size);
              
patch_height = vPatch_size(1);
patch_width  = vPatch_size(2);
              
mSD = randn(vSignature_size);

max_iterations = size(mTrain, 2);
vF            = zeros(1, floor(max_iterations / 200));
vMean_support = zeros(1, floor(max_iterations / 200));
k             = 1;
for ii = 1 : max_iterations

    %% OMP:
    vTrain = mTrain(:,ii);
%     mD = im2col(mSD, vPatch_size);
    mD = mSD(mD_idx); % for speed...
    vW = sqrt( sum(mD.^2, 1) );
    mA = bsxfun(@rdivide, mD, vW);
    vX = omp2(mA, vTrain, [], threshold);
    vX = vX ./ vW';
    vR = mD * vX - vTrain;
    
    %% Gradient:
    mSG = zeros(size(mSD));
    for jj = find(vX)'
        vG1    = vX(jj) * vR;
        [y, x] = ind2sub(size(mSD) - vPatch_size + 1, jj);
        vY_ind = y : y + patch_height - 1;
        vX_ind = x : x + patch_width - 1;
        vG1    = reshape(vG1, vPatch_size);
        mSG(vY_ind,vX_ind) = mSG(vY_ind,vX_ind) + vG1;
    end

    %% Update:
    mu  = .001;
    mSD = mSD - mu * mSG;
    
    if mod(ii, 200) == 1 && 0
        mD = mSD(mD_idx);
        vW = sqrt( sum(mD.^2, 1) );
        mA = bsxfun(@rdivide, mD, vW);
        mG = mA' * mA;
        mX = omp2(mA' * mTrain, sum(mTrain.*mTrain), mG, threshold);
        
%         mX = omp(mA' * mTrain, mG, cardinality);
        
        mR    = mA * mX - mTrain;
        vF(k) = sqrt( mean( mean((mR).^2, 1) ) );
        vMean_support(k) = mean( sum(~~mX, 1), 2);
        k     = k + 1;
    end
  
end

end
              
              