function [is_period_included] = fnFilterPeriodsWithCriteria(active_processing, included_epochs, included_states)
%fnFilterPeriodsWithCriteria Period will be included if it belongs to one of the specified epoch types AND one of the specified included states
%   Detailed explanation goes here

% included_epochs: a cell array of epoch types to include. Period will be included if it belongs to one of the specified epoch types AND one of the specified included states
    % e.g. {'pre_sleep', 'post_sleep'}
% included_states: a cell array of behavioral state types to include.
    % e.g. {'rem'}
    
% NOTE: If an empty list is passed, all (either epochs or periods) will be included, and the result will only be filtered on the non-empty criteria.
    
    num_of_behavioral_state_periods = height(active_processing.behavioral_periods_table);
%     is_period_included = logical(ones([num_of_behavioral_state_periods 1]));
    
    %% Check for exclusion by epoch:
    if exist('included_epochs','var') & ~isempty(included_epochs)
%         is_epoch_included = false([num_of_behavioral_state_periods 1]);
%         curr_conditions.is_behavioral_epoch_type = (@(compare_type) (active_processing.behavioral_periods_table.behavioral_epoch == compare_type));
%         temp.curr_is_included = cellfun(curr_conditions.is_behavioral_epoch_type, included_epochs, 'UniformOutput', false);
%         % Iterate through all the conditions:
%         for i = 1:length(temp.curr_is_included)
% %             is_period_included = is_period_included & temp.curr_is_included{i};
%             is_epoch_included = is_epoch_included | temp.curr_is_included{i}; % Should be or (|) as it must meet one of the requirements for epoch to be included.
%         end

       [is_epoch_included] = BehavioralEpoch.filter(active_processing.behavioral_periods_table.behavioral_epoch, included_epochs);

    else
        % When empty, return true for all of this type
        is_epoch_included = true([num_of_behavioral_state_periods 1]);
    end


    %% Check for exclusion by sleep state:
    
    if exist('included_states','var') & ~isempty(included_states)
%         is_state_included = false([num_of_behavioral_state_periods 1]);
%         curr_conditions.is_behavioral_state_type = (@(compare_type) (active_processing.behavioral_periods_table.type == compare_type));
%         temp.curr_is_included = cellfun(curr_conditions.is_behavioral_state_type, included_states, 'UniformOutput', false);
%         for i = 1:length(temp.curr_is_included)
% %             is_period_included = is_period_included & temp.curr_is_included{i};
%             is_state_included = is_state_included | temp.curr_is_included{i};
%         end

        [is_state_included] = BehavioralState.filter(active_processing.behavioral_periods_table.type, included_states);
    else
        % When empty, return true for all of this type
        is_state_included = true([num_of_behavioral_state_periods 1]);
    end


    

    
    % Finally, for the period to be included in general it must meet both the epoch and state requirements, so it should be logical AND
    is_period_included = is_epoch_included & is_state_included;

end

