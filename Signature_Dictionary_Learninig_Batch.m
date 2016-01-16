function [mSD, mSD0, vF1, vF2] = Signature_Dictionary_Learninig_Batch(...
                  vSignature_size, mTrain, vPatch_size, cardinality)

vD_idx = 1 : prod(vSignature_size);
mD_idx = reshape(vD_idx, vSignature_size);
mD_idx = im2col(mD_idx, vPatch_size);
              
patch_height = vPatch_size(1);              
patch_width  = vPatch_size(2);
              
mSD      = randn(vSignature_size);
mSD0     = mSD;
D_length = (size(mSD,1) - patch_height + 1)^2;

max_iterations = 200;
vF1 = zeros(1, max_iterations);
vF2 = zeros(1, max_iterations);

for ii = 1 : max_iterations
    
    %% OMP:
%     mD = im2col(mSD, vPatch_size);
    mD = mSD(mD_idx);
    vW = sqrt( sum(mD.^2, 1) );
    mA = bsxfun(@rdivide, mD, vW);
    mG = mA' * mA;
    mX = omp(mA' * mTrain, mG, cardinality);
    mX = bsxfun(@rdivide, mX, vW');
   
    mR      = mD * mX - mTrain;
    vF1(ii) = sqrt( mean( mean((mR).^2, 1) ) );
    
    %% Gradient:
    mSG = zeros(size(mSD));
    for jj = 1 : D_length
        vP         = find(mX(jj,:));
        mG1        = bsxfun(@times, mR(:,vP), full(mX(jj,vP)));
        mG1        = sum(mG1, 2);
        [y, x]     = ind2sub(size(mSD) - vPatch_size + 1, jj);
        vY         = y : y + patch_height - 1;
        vX         = x : x + patch_width - 1;
        mG1        = reshape(mG1, vPatch_size);
        mSG(vY,vX) = mSG(vY,vX) + mG1;
    end

    %% Step Size:
    mu    = .001;
    mSDk  = mSD - mu * mSG;
%     mD    = im2col(mSDk, vPatch_size );
    mD    = mSDk(mD_idx);
    mR    = mTrain - mD * mX;
    f_new = sqrt( mean( mean((mR).^2, 1) ) );
    f_old = inf;
    while f_new < f_old
        f_old = f_new;
        mu    = 0.9 * mu;
        mSDk  = mSD - mu * mSG;
%         mD    = im2col(mSDk, vPatch_size);
        mD    = mSDk(mD_idx);
        mR    = mTrain - mD * mX;
        f_new = sqrt( mean( mean((mR).^2, 1) ) );
    end
    vF2(ii) = f_old;
    
    %% Update:
    mu  = mu / 0.9;
    mSD = mSD - mu * mSG;
    disp([num2str(f_old),' ', num2str(mu)])
   

end


end
              
              