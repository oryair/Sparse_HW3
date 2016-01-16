function [mSD, mSD0, vF1] = Signature_Dictionary_Learninig_On_Line(...
                  signature_size, mTrain, patch_size, cardinality)

patch_height = patch_size(1);              
patch_width  = patch_size(2);
              
mSD      = randn(signature_size);
mSD0     = mSD;

max_iterations = size(mTrain, 2);
vF1 = zeros(1, floor(max_iterations / 200));
k   = 1;

for ii = 1 : max_iterations
    
    %% OMP:
    vTrain = mTrain(:,ii);
    mD = im2col(mSD, patch_size);
    vW = sqrt( sum(mD.^2, 1) );
    mA = bsxfun(@rdivide, mD, vW);
%     mG = mA' * mA;
%     vX = omp(mA' * vTrain, mG, cardinality);
    vX = omp(mA, vTrain, [], cardinality);
    vX = vX ./ vW';
    vR = mD * vX - vTrain;
    
    %% Gradient:
    mSG = zeros(size(mSD));
    for jj = find(vX)'
        vG1    = vX(jj) * vR;
        [y, x] = ind2sub(size(mSD) - patch_size + 1, jj);
        vY_ind = y : y + patch_height - 1;
        vX_ind = x : x + patch_width - 1;
        vG1    = reshape(vG1, patch_size);
        mSG(vY_ind,vX_ind) = mSG(vY_ind,vX_ind) + vG1;
    end

    %% Update:
    mu  = .001;
    mSD = mSD - mu * mSG;
    
    %% Check:
    if mod(ii, 200) == 1
        mD = im2col(mSD, patch_size);
        vW = sqrt( sum(mD.^2, 1) );
        mA = bsxfun(@rdivide, mD, vW);
        mG = mA' * mA;
        mX = omp(mA' * mTrain, mG, cardinality);
        mR = mA * mX - mTrain;
        vF1(k) = sqrt( mean( mean((mR).^2, 1) ) );
        k = k + 1;
    end
   
end

end
              
              