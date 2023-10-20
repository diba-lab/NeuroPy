function [includedCellIDs, unitSpikeCells, unitFlatIndicies] = fnFlatSpikesToUnitCells(flatSpikeTimes, flatSpikeUnitIDs, includeIndicies)
% fnFlatSpikesToUnitCells - Converts a flat spikes representation (where each spike is represented by a time in flatSpikeTimes and a corresponding cell ID in flatSpikeUnitIDs) to a cell array of the spikes (grouped by unit).
% Detailed explanation goes here
% 
% Syntax:  
%     [includedCellIDs, unitSpikeCells, unitFlatIndicies] = fnFlatSpikesToUnitCells(flatSpikeTimes, flatSpikeUnitIDs, includeIndicies)
% 
% Inputs:
%    flatSpikeTimes - Description
%    flatSpikeUnitIDs - Description
%    includeIndicies - Description
% 
% Outputs:
%    includedCellIDs - Description
%    unitSpikeCells - Description
%    unitFlatIndicies - Description
% 
% Author: Pho Hale
% PhoHale.com 
% email: halechr@umich.edu
% Created: 29-Oct-2021 ; Last revision: 29-Oct-2021 

% ------------- BEGIN CODE --------------

    if size(flatSpikeTimes) ~= size(flatSpikeUnitIDs) 
        error('flatSpikeTimes is not the same size as flatSpikeUnitIDs!')
    end
    if ~exist('includeIndicies','var')
        includeIndicies = false;
    end
    
    includedCellIDs = unique(flatSpikeUnitIDs);
    numOutputCellIDs = length(includedCellIDs);
    unitSpikeCells = cell(numOutputCellIDs, 1);
    if includeIndicies
        unitFlatIndicies = cell(numOutputCellIDs, 1);
    else
        unitFlatIndicies = {};
    end
    
    for i = 1:numOutputCellIDs
         if includeIndicies
            unitFlatIndicies{i} = find(includedCellIDs(i) == flatSpikeUnitIDs);
            unitSpikeCells{i} = flatSpikeTimes(unitFlatIndicies{i});
         else
            unitSpikeCells{i} = flatSpikeTimes(includedCellIDs(i) == flatSpikeUnitIDs);
         end
    end
end
% ------------- END OF CODE --------------