function [is_pair_included, original_pair_index] = fnFilterPairsWithCriteria(general_results, included_unit_indicies)
% fnFilterPairsWithCriteria - One line description of what the function performs
% Detailed explanation goes here
% 
% Syntax:  
%     [is_pair_included, original_pair_index] = fnFilterPairsWithCriteria(general_results, included_unit_indicies)
% 
% Input:
%    general_results - Description
%    included_unit_indicies - Description
% 
% Outputs:
%    output1 - Description
%    output2 - Description
% 
% Author: Pho Hale
% PhoHale.com 
% email: halechr@umich.edu
% Created: 29-Oct-2021 ; Last revision: 29-Oct-2021 

% ------------- BEGIN CODE --------------


%fnFilterPairsWithCriteria Filters the pairs of unit indicies
	% included_unit_indicies: an array of units to include.
    
    
	%% Outputs:
	% is_pair_included: a numPairs x length(included_unit_indicies) matrix where each column specifies whether that pair is included for this index.
    % original_pair_index: a numUnits x length(included_unit_indicies) matrix where each column contains the original linear pair indicies for the corresponding index.
	num_included_unit_indicies = length(included_unit_indicies);
	num_original_units = size(general_results.indicies.reverse_lookup_unique_electrode_pairs, 2);
    num_original_pairs = general_results.indicies.num_unique_pairs;
	%% Pre-allocate:
    is_pair_included = logical(zeros([num_original_pairs num_included_unit_indicies]));
	original_pair_index = zeros([num_original_units num_included_unit_indicies]);
	for i = 1:length(included_unit_indicies)
		temp.curr_unit_index = included_unit_indicies(i);
		
		% Get all elements from the lookup table corresponding to this index.
		temp.curr_found_lin_indicies = general_results.indicies.reverse_lookup_unique_electrode_pairs(temp.curr_unit_index, :); % 1x126 double
		% These are the indicies that belong to that item in the pair array:
        temp.excluded_indicies = find(temp.curr_found_lin_indicies < 1);
        temp.curr_found_lin_indicies(temp.excluded_indicies) = temp.curr_unit_index; % temporarily fill it in with a valid index (referring to row 1), but then replace it afterwards with NaN
    
		
		is_pair_included(temp.curr_found_lin_indicies, i) = 1; % Uses these indicies to set those pair elements to true;
		original_pair_index(:, i) = temp.curr_found_lin_indicies;
	end
% 	temp.curr_pairs_indicies = find(filter_config.filter_active_units);
%     filter_config.filter_active_pairs =  ismember(general_results.indicies.unique_electrode_pairs(:,1), temp.curr_pairs_indicies) & ismember(general_results.indicies.unique_electrode_pairs(:,2), temp.curr_pairs_indicies);
%     filter_config.filter_active_pair_values = general_results.indicies.unique_electrode_pairs(filter_config.filter_active_pairs, :);
end


% ------------- END OF CODE --------------
