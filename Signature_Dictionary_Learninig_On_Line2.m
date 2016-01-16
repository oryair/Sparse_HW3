function mSD = Signature_Dictionary_Learninig_On_Line2(...
                  signature_size, mTrain, patch_size, threshold)

patch_height = patch_size(1);              
patch_width  = patch_size(2);
              
mSD = randn(signature_size);

max_iterations = size(mTrain, 2);

for ii = 1 : max_iterations
    
    %% OMP:
    vTrain = mTrain(:,ii);
    mD = im2col(mSD, patch_size);
    vW = sqrt( sum(mD.^2, 1) );
    mA = bsxfun(@rdivide, mD, vW);
    vX = omp2(mA, vTrain, [], threshold);
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
  
end

end
              
              