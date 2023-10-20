function Y = nchoose2(X)
% NCHOOSE2 - all combinations of two elements
%   Y = NCHOOSE2(X) returns all combinations of two elements of the array X.
%   It is the very fast, vectorized version of NCHOOSEK(X,2).  X can be any
%   type of array. When X is a scalar integer > 1, it returns X*(X-1)/2.
%
%   Examples:
%      nchoose2([10 20 30 40])
%      % -> 10    20
%      %    10    30
%      %    10    40
%      %    20    30
%      %    20    40
%      %    30    40
%
%      nchoose2({'a','b','c','d','e'})
%      % -> 'a'  'b'
%      %    'a'  'c'
%      %      ...
%      %    'c'  'e'
%      %    'd'  'e'
%
%   See also NCHOOSEK, PERMS, NEXTPERM
%            PERMN, NCHOOSE, ALLCOMB (on the File Exchange)

% version 3.2 (feb 2019)
% (c) Jos van der Geest
% email: samelinoa@gmail.com
% http://www.mathworks.uk/matlabcentral/fileexchange/authors/10584

% History
% 1.0 sep 2007 - created, for faster solution of nchoosek(x,2)
% 2.0 may 2008 - inspired to put on the FEX, by submission #20110 by S.
%                Scaringi, and review by John D'Errico
%              - optimized engine, added extensive help and comments
% 2.1 jun 2008 - catch error when X has less than two elements
%                (error pointed out by Urs Schwarz)
% 2.2 dec 2017 - updated for newer ML releases
% 3.0 feb 2018 - when X is a scalar integer, returns (X*(X-1)/2), like
%                nchoosek(X,2)
% 3.1 mar 2018 - updated bug for scalar input
% 3.2 feb 2019 - modernised, updated comments 

N = numel(X) ;
if N==1
    if isnumeric(X) && X > 1 && X == fix(X)
        Y = X*(X-1)/2 ;
    else
        error('For scalar input, N should be an integer > 1.') ;
    end    
elseif N == 2
    % only two elements
    Y = reshape(X, 1, 2) ; % output is a row vector
else % N > 2
    % by creating an (N*(N-1)/2)-by-2 index matrix I the output can be
    % retrieved directly. This index matrix I equals nchoosek(1:N, 2).
    % N is the number of elements if X. We create I step-by-step using
    % left-hand indexing.    
                                  % Example for N = 4 ->
    V  = N-1:-1:2 ;               % V : 3 2
    R = cumsum([1 V], 2) ;        % R : 1 4 6

    % Step 1 - create I, filling the two columns (c1 and c2)
    I(R,2) = [0 -V] + 1 ;         % -> c1: 0  0  0  0  0  0
                                  %    c2: 1  0  0 -2  0 -1
    % Step 2                              
    I(R,1) = 1 ;                  % -> c1: 1  0  0  1  0  1
                                  %    c2: 1  0  0 -2  0 -1
    % Step 3
    I(:,2) = I(:,2) + 1 ;         % -> c1: 1  0  0  1  0  1
                                  %    c2: 2  1  1 -1  1  0
    % Step 4
    I = cumsum(I, 1) ;            % -> c1: 1  1  1  2  2  3
                                  %    c2: 2  3  4  3  4  4
    % Now we use I to index directly into X to create the output
    Y = X(I) ;                    
end

%   Notes:
%   - NCHOOSE2(X) is much faster than NCHOOSEK(X,2). It is also faster than
%     another solution ("nCtwo", FEX # 20110, May 28th 2008, Simone
%     Scaringi), especially for smaller arrays. It is also more memory
%     efficient than a solution based on a suggestion by John D'Errico in
%     his review of FEX #20110:
%        [I,J] = find(tril(ones(numel(x)),-1));
%        y = x([J(:) I(:)]);
%     The latter solution is a little faster for shorter vectors, but
%     slower and memory consuming for larger vectors. Moreover, it may
%     require a call to  sortrows to get the same order as nchoosek.
%   - specifying the dimension for cumsum is slightly faster