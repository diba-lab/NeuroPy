function [names] = fnGetMatFileVariableNames(matFilePath)
% fnGetMatFileVariableNames - Instantly returns the list of variable names in the .mat file specified by matFilePath (without having to load it at all)
% Detailed explanation goes here
% 
% Syntax:  
%     [names] = fnGetMatFileVariableNames(matFilePath)
% 
% Input:
%    matFilePath - The file path to the .mat file
% 
% Outputs:
%    names - Cell array of variable names
%    extended_info - 3x1 struct array returned by 'whos' command
% 
% Author: Pho Hale
% PhoHale.com 
% email: halechr@umich.edu
% Created: 30-Oct-2021 ; Last revision: 30-Oct-2021 

% ------------- BEGIN CODE --------------
   
    if ~exist(matFilePath,"file")
        names = {};
    else
        m = matfile(matFilePath);
    %     extended_info = whos(m);
        names = who(m);
    end
end


% ------------- END OF CODE --------------
