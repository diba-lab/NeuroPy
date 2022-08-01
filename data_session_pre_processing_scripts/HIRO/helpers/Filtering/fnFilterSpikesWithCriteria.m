function [is_target_entry_included] = fnFilterSpikesWithCriteria(filter_target, included_epochs, included_states, target_options)
% fnFilterSpikesWithCriteria - One line description of what the function or script performs
% Detailed explanation goes here
% 
% Syntax:  
%     [is_target_entry_included] = fnFilterSpikesWithCriteria(filter_target, included_epochs, included_states, target_options)
% 
% Inputs:
%    filter_target - contains the target with the spiking data to be filtered
%    included_epochs - a cell array of epoch types to include. Period will be included if it belongs to one of the specified epoch types AND one of the specified included states
         % e.g. {'pre_sleep', 'post_sleep'}
        % NOTE: If an empty list is passed, all (either epochs or periods) will be included, and the result will only be filtered on the non-empty criteria.
%    included_states - a cell array of behavioral state types to include.
        % e.g. {'rem'}
        % NOTE: If an empty list is passed, all (either epochs or periods) will be included, and the result will only be filtered on the non-empty criteria.
%    target_options - Description
% 
% Outputs:
%    is_target_entry_included - Description
% 


% Author: Pho Hale
% PhoHale.com 
% email: halechr@umich.edu
% Created: 29-Oct-2021 ; Last revision: 29-Oct-2021 

% ------------- BEGIN CODE --------------

% included_ripple_index
% included_cell_types
% active_processing.spikes.time(plot_outputs.filter_active_units)
% filtered_spike_times = cellfun(@(spike_times, is_spike_ripple) spike_times(is_spike_ripple), spikesTable.time, spikesTable.isRippleSpike, 'UniformOutput', false); %% Filtered to only show the ripple spikes
    if ~exist('filter_target','var') || ~istabular(filter_target)
        error('fnFilterSpikesWithCriteria: The filter_target must currently be a table! Aborting.')
    end
    if ~exist('target_options','var')
        target_options = struct();
    end
    
    % For compatibility with the old naming conventions, you might need to use:
%     target_options.behavioral_states_variable_name = 'type';
    target_options = fnAddDefaultOptionalArgs({'behavioral_states_variable_name', 'behavioral_epochs_variable_name'}, ...
        {'type', 'behavioral_epoch'}, ...
        target_options);
    
    num_of_target_entries = height(filter_target);
    
    %% Check for exclusion by epoch:
    if exist('included_epochs','var') & ~isempty(included_epochs)
       [is_epoch_included] = BehavioralEpoch.filter(filter_target.(target_options.behavioral_epochs_variable_name), included_epochs);
    else
        % When empty, return true for all of this type
        is_epoch_included = true([num_of_target_entries 1]);
    end
    %% Check for exclusion by sleep state:
    
    if exist('included_states','var') & ~isempty(included_states)
        [is_state_included] = BehavioralState.filter(filter_target.(target_options.behavioral_states_variable_name), included_states);
    else
        % When empty, return true for all of this type
        is_state_included = true([num_of_target_entries 1]);
    end
    % Finally, for the period to be included in general it must meet both the epoch and state requirements, so it should be logical AND
    is_target_entry_included = is_epoch_included & is_state_included;
end


% ------------- END OF CODE --------------
