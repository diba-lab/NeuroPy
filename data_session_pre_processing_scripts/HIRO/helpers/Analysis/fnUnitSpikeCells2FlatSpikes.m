function [flatSpikeTimes, flatSpikeUnitIDs] = fnUnitSpikeCells2FlatSpikes(unitSpikeCells, includedCellIDs)
%fnUnitSpikeCells2FlatSpikes Converts a cell array of the spikes (grouped by unit) to a flat spikes representation (where each spike is represented by a time in flatSpikeTimes and a corresponding cell ID in flatSpikeUnitIDs)
% Inverse of fnFlatSpikesToUnitCells

    num_unit_cells = length(unitSpikeCells);
    
    if ~exist('includedCellIDs','var')
        % If no IDs provided, generate IDs 1:N for each unit
        includedCellIDs = 1:num_unit_cells;
        if size(unitSpikeCells) ~= size(includedCellIDs) 
            % Try the transpose
            includedCellIDs = includedCellIDs';
        end
    end

    if size(unitSpikeCells) ~= size(includedCellIDs) 
        % Try the transpose
%         unitSpikeCells = unitSpikeCells';
%         includedCellIDs = includedCellIDs';
        % If it's still not the correct size, give up
%         if size(unitSpikeCells) ~= size(includedCellIDs)
            error('unitSpikeCells is not the same size as includedCellIDs!')
%         end
    end
    
    flatSpikeTimes = [];
    flatSpikeUnitIDs = [];

    % Closed-form vectorized flattening for non-equal sized cells
    flatSpikeTimes = [unitSpikeCells{:}];

    for i = 1:num_unit_cells
%         flatSpikeTimes = [flatSpikeTimes; unitSpikeCells{i}]; %Previously used this method to build up the array, but had issues with vertcat 
        flatSpikeUnitIDs = [flatSpikeUnitIDs; repmat(includedCellIDs(i),[length(unitSpikeCells{i}) 1])];
    end

    [flatSpikeTimes, sort_indicies] = sort(flatSpikeTimes);
    flatSpikeUnitIDs = flatSpikeUnitIDs(sort_indicies);
end

