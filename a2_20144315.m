function [rmsvars lowndx rmstrain rmstest] = a2_00000000
% [RMSVARS LOWNDX RMSTRAIN RMSTEST]=A3 finds the RMS errors of
% linear regression of the data in the file "AIRQUALITY.CSV" by treating
% each column as a vector of dependent observations, using the other
% columns of the data as observations of independent varaibles. The
% individual RMS errors are returned in RMSVARS and the index of the
% smallest RMS error is returned in LOWNDX. For the variable that is
% best explained by the other variables, a 5-fold cross validation is
% computed. The RMS errors for the training of each fold are returned
% in RMSTEST and the RMS errors for the testing of each fold are
% returned in RMSTEST.
%
% INPUTS:
%         none
% OUTPUTS:
%         RMSVARS  - 1xN array of RMS errors of linear regression
%         LOWNDX   - integer scalar, index into RMSVALS
%         RMSTRAIN - 1x5 array of RMS errors for 5-fold training
%         RMSTEST  - 1x5 array of RMS errors for 5-fold testing

    filename = 'A1.csv';
    [rmsvars lowndx] = a2q1(filename);
    [rmstrain rmstest] = a2q2(filename, 4)

end

function [rmsvars lowndx] = a2q1(filename)
% [RMSVARS LOWNDX]=A2Q1(FILENAME) finds the RMS errors of
% linear regression of the data in the file FILENAME by treating
% each column as a vector of dependent observations, using the other
% columns of the data as observations of independent varaibles. The
% individual RMS errors are returned in RMSVARS and the index of the
% smallest RMS error is returned in LOWNDX. 
%
% INPUTS:
%         FILENAME - character string, name of file to be processed;
%                    assume that the first row describes the data variables
% OUTPUTS:
%         RMSVARS  - 1xN array of RMS errors of linear regression
%         LOWNDX   - integer scalar, index into RMSVALS

    % Read file and set all negative values to NaN
    Amat = csvread(filename, 1, 1);
    Amat(Amat<0) = NaN;
    
    % Fill missing values with a linear interpolation model
    Amat = fillmissing(Amat, 'linear');
    
    sz = size(Amat);
    
    % Standardize the matrix
    Amat_norm = normalize(Amat);
    
    % Find RMSE for linear regression
    rmsvars = zeros(1, size(Amat_norm, 2));
    for i=1 : size(Amat, 2)
        Xmat = Amat; % Line with no intercept term
        Xmat(:, i)=[]; % Create Xmat
        cvec = Amat_norm(:, i);
        uval = Xmat\cvec;
        rmsvars(i) = rms(cvec - Xmat*uval); 
    end
    
    % Find lowest rms error
    lowndx = find(rmsvars == min(rmsvars));

    % Finding regression in unstandardized variables
    unst_rmsvars = zeros(1, size(Amat, 2));
    for i=1 : size(Amat, 2)
        Xmat = [Amat ones(size(Amat, 1), 1)]; % Line with intercept term
        Xmat(:, i)=[];
        cvec = Amat(:, i);
        uval = Xmat\cvec;
        unst_rmsvars(i) = rms(cvec - Xmat*uval);
    end
    
    % Find lowest rms error
    unst_lowndx = find(unst_rmsvars == min(unst_rmsvars));
    
    % Compare standardized vs unstandardized.
    rmsvars;
    unst_rmsvars;

    % plot every 90 values
    plot_v = Amat(:, lowndx);
    t = 1:numel(plot_v);
    plot(t(90:90:end),plot_v(90:90:end));
    title('Linear Regression of Dependent Variable hourly averaged sensor response')
    xlabel('Time (arbitrary units)')
    ylabel('Sensor Response')
    
end
function [rmstrain rmstest] = a2q2(filename,lowndx)
% [RMSTRAIN RMSTEST]=A3Q2(LOWNDX) finds the RMS errors of 5-fold
% cross-validation for the variable LOWNDX of the data in the file
% FILENAME. The RMS errors for the training of each fold are returned
% in RMSTEST and the RMS errors for the testing of each fold are
% returned in RMSTEST.
%
% INPUTS:
%         FILENAME - character string, name of file to be processed;
%                    assume that the first row describes the data variables
%         LOWNDX   - integer scalar, index into the data
% OUTPUTS:
%         RMSTRAIN - 1x5 array of RMS errors for 5-fold training
%         RMSTEST  - 1x5 array of RMS errors for 5-fold testing

    % Read file and set all negative values to NaN
    Amat = csvread(filename, 1, 0);
    Amat(Amat<0) = NaN;
    
    % Fill missing values with a linear interpolation model
    Amat = fillmissing(Amat, 'linear');
    sz = size(Amat);
    
    % Find yvec and Xmat from unstandardized data
    yvec = Amat(:, lowndx);
    Xmat = Amat;
    Xmat(:, lowndx)=[];
    
    % Call mykfold function to compute k-fold cross validation
    [rmstrain rmstest] = mykfold(Xmat, yvec, 5);

end

function [rmstrain,rmstest]=mykfold(Xmat, yvec, k_in)
% [RMSTRAIN,RMSTEST]=MYKFOLD(XMAT,yvec,K) performs a k-fold validation
% of the least-squares linear fit of yvec to XMAT. If K is omitted,
% the default is 5.
%
% INPUTS:
%         XMAT     - MxN data vector
%         yvec     - Mx1 data vector
%         K        - positive integer, number of folds to use
% OUTPUTS:
%         RMSTRAIN - 1xK vector of RMS error of the training fits
%         RMSTEST  - 1xK vector of RMS error of the testing fits

    % Problem size
    M = size(Xmat, 1);

    % Set the number of folds; must be 1<k<M
    if nargin >= 3 & ~isempty(k_in)
        k = max(min(round(k_in), M-1), 2);
    else
        k = 5;
    end

    % Initialize the return variables
    rmstrain = zeros(1, k);
    rmstest  = zeros(1, k);
    
    % Random permutation of rows
    random_row = randperm(size(Xmat, 1))';
    
    indices = zeros(M, 1);
    i = 1;
    
    % Generate random indices
    for ix=1:size(Xmat, 1)
        indices(random_row(ix)) = i;
        i = i+1;
        if i > k
            i = 1;
        end
    end
    
    % Process each fold
    for ix=1:k
        
        % Set test and train indices
        test = (indices == ix);
        train = ~test;
        
        % Set test and train data with intercept term
        xmat_train = [Xmat(train, :) ones(size(Xmat(train, :), 1), 1)];
        xmat_test = [Xmat(test, :) ones(size(Xmat(test, :), 1), 1)];

        
        yvec_train = yvec(train, :);
        yvec_test = yvec(test, :);

        % Compute wvec
        wvec = xmat_train\yvec_train;
        
        % Find RMSE for each fold
        rmstrain(ix) = rms(xmat_train*wvec - yvec_train);
        rmstest(ix)  = rms(xmat_test*wvec  - yvec_test);
    end
    
    % Calculate Mean and Variance
    mean_train = mean(rmstrain);
    mean_test = mean(rmstest);
    var_train = std(rmstrain);
    var_test = std(rmstest);
    

end
