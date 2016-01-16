function [mSD, mSD0, vF1, vF2] = Signature_Dictionary_Learninig_Batch(...
                  signature_size, mTrain, patch_size, cardinality)

patch_height = patch_size(1);              
patch_width  = patch_size(2);
              
mSD      = randn(signature_size);
mSD0     = mSD;
D_length = (size(mSD,1) - patch_height + 1)^2;

max_iterations = 200;
vF1 = zeros(1, max_iterations);
vF2 = zeros(1, max_iterations);

for ii = 1 : max_iterations
    
    %% OMP:
    mD = im2col(mSD, patch_size);
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
        [y, x]     = ind2sub(size(mSD) - patch_size + 1, jj);
        vY         = y : y + patch_height - 1;
        vX         = x : x + patch_width - 1;
        mG1        = reshape(mG1, patch_size);
        mSG(vY,vX) = mSG(vY,vX) + mG1;
    end

    %% Step Size:
    mu    = .001;
    mSDk  = mSD - mu * mSG;
    mD    = im2col(mSDk, patch_size );
    mR    = mTrain - mD * mX;
    f_new = sqrt( mean( mean((mR).^2, 1) ) );
    f_old = inf;
    while f_new < f_old
        f_old = f_new;
        mu    = 0.9 * mu;
        mSDk  = mSD - mu * mSG;
        mD    = im2col(mSDk, patch_size);
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
              
              