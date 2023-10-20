function [curr_flattened_table] = fnFlattenCellsToContents(varargin)
%FNFLATTENCELLSTOCONTENTS accepts a single table or one or more cell arrays of length N, with each cell containing items with the same datatype but potentially different sizes.
%   Produces a flattened representation of all the contained items, returned in the table `curr_flattened_table`

%% Output:
% curr_flattened_table: a table

%% Example:
    % % Test single cell array input:
    % [curr_flattened_table] = fnFlattenCellsToContents(active_processing.spikes.time, active_processing.spikes.isRippleSpike, active_processing.spikes.RippleIndex);
    % % Test multiple cell array inputs:
    % [curr_flattened_table] = fnFlattenCellsToContents(active_processing.spikes.time, active_processing.spikes.isRippleSpike, active_processing.spikes.RippleIndex);
    % 
    % % Test table input version:
    % curr_cell_table = table(active_processing.spikes.time, ...
    %         active_processing.spikes.isRippleSpike, ...
    %         active_processing.spikes.RippleIndex, ...
    %         'VariableNames', {'time','isRippleSpike','RippleIndex'});
    % [curr_flattened_table] = fnFlattenCellsToContents(curr_cell_table);

    numFnInputs = nargin;
    
    if numFnInputs == 0
        error('fnFlattenCellsToContents(...) called with no arguments!');
    elseif ((numFnInputs == 1) & istabular(varargin{1}))
    %     if istabular(varargin{1})
            % A table object
            curr_cell_table = varargin{1};
            first_cell_array = curr_cell_table{:,1};
            num_cell_arrays_to_flatten = width(curr_cell_table);
    else
        % Otherwise more than one input, so we assume they're all cell arrays
        curr_cell_table = table(varargin{:});
        first_cell_array = curr_cell_table{:,1};
        num_cell_arrays_to_flatten = width(curr_cell_table);
    end
    
    num_cells = length(first_cell_array);
    cell_content_counts = cellfun(@length, first_cell_array);
    cell_indicies = 1:num_cells;
    
    % The total number of output items for the flattened array
    flattened_total_item_count = sum(cell_content_counts, 'all');
    
    % flattened_UnitIDs: the unit the original entry belongs to:
    flattened_UnitIDs = repelem(cell_indicies, cell_content_counts); % the second argument specifies how many times to repeat each item in the first array
    
    % Now flatten each table variable:
    % Closed-form vectorized flattening for non-equal sized cells
    curr_flattened_table = table('Size', [flattened_total_item_count, (1 + num_cell_arrays_to_flatten)], 'VariableNames', ['flattened_UnitIDs', curr_cell_table.Properties.VariableNames], 'VariableTypes', ['double', varfun(@class, curr_cell_table, 'OutputFormat', 'cell')]);
    % Add the flattened unit id's as the first column
    curr_flattened_table.flattened_UnitIDs = flattened_UnitIDs';
    
    for var_idx = 1:num_cell_arrays_to_flatten
        curr_variable_name = curr_cell_table.Properties.VariableNames{var_idx};
        curr_flattened_table.(curr_variable_name) = [curr_cell_table{:,var_idx}{:}]';
    end
    
    % Sort the output table by the time column:
    if isVariable(curr_flattened_table, 'time')
        curr_flattened_table = sortrows(curr_flattened_table,'time','ascend');
    else
        % Otherwise sort on whatever column 2 is (since column 1 contains the generated flattened_UnitIDs
        curr_flattened_table = sortrows(curr_flattened_table, 2, 'ascend');
    end
    
end

