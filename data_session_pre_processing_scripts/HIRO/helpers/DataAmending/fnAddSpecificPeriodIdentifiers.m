function [periods_table] = fnAddSpecificPeriodIdentifiers(periods_table)
% fnAddSpecificPeriodIdentifiers: From a table containing the behavioral periods and their corresponding states, generates label columns for each categorical state column;
% currently hardcoded for .type and .behavioral_epoch columns.
% periods_table: a active_processing.behavioral_periods_table style table
%% Usage Example:
% [active_processing.behavioral_periods_table] = fnAddSpecificPeriodIdentifiers(active_processing.behavioral_periods_table);

if ~exist('periods_table','var')
    periods_table_workspace_variable_name = 'active_processing.behavioral_periods_table';
    periods_table = evalin('caller', periods_table_workspace_variable_name);
%     periods_table = active_processing.behavioral_periods_table;
    table_was_loaded_from_workspace = true;

else
    table_was_loaded_from_workspace = false;
end




%% Example Input Table:
%% size(active_processing.behavioral_periods_table) = 668     3
% type      behavioral_epoch
% 'nrem'	'pre_sleep'
% 'quiet'	'pre_sleep'
% 'nrem'	'pre_sleep'
% 'rem'	'pre_sleep'
% 'nrem'	'pre_sleep'
% 'rem'	'pre_sleep'
% 'nrem'	'pre_sleep'
% 'rem'	'pre_sleep'
% 'nrem'	'pre_sleep'
% 'rem'	'pre_sleep'
% 'nrem'	'pre_sleep'
% 'quiet'	'pre_sleep'
% 'nrem'	'pre_sleep'
% 'quiet'	'pre_sleep'
% 'active'	'pre_sleep'
% ...       ...

%% Updated Output Table: 
%% size(active_processing.behavioral_periods_table) = 668     5
% type      behavioral_epoch    type_label  behavioral_epoch_label
% 'nrem'	'pre_sleep'	'nrem_1'	'pre_sleep_1'
% 'quiet'	'pre_sleep'	'quiet_1'	'pre_sleep_2'
% 'nrem'	'pre_sleep'	'nrem_2'	'pre_sleep_3'
% 'rem'	'pre_sleep'	'rem_1'	'pre_sleep_4'
% 'nrem'	'pre_sleep'	'nrem_3'	'pre_sleep_5'
% 'rem'	'pre_sleep'	'rem_2'	'pre_sleep_6'
% 'nrem'	'pre_sleep'	'nrem_4'	'pre_sleep_7'
% 'rem'	'pre_sleep'	'rem_3'	'pre_sleep_8'
% 'nrem'	'pre_sleep'	'nrem_5'	'pre_sleep_9'
% 'rem'	'pre_sleep'	'rem_4'	'pre_sleep_10'
% 'nrem'	'pre_sleep'	'nrem_6'	'pre_sleep_11'
% 'quiet'	'pre_sleep'	'quiet_2'	'pre_sleep_12'
% 'nrem'	'pre_sleep'	'nrem_7'	'pre_sleep_13'
% 'quiet'	'pre_sleep'	'quiet_3'	'pre_sleep_14'
% 'active'	'pre_sleep'	'active_1'	'pre_sleep_15'
% ...       ...         ...         ...



% periods_table = active_processing.behavioral_periods_table;
% period_entries = periods_table.type;

[period_label_output] = subfnBuildUpdatedTable(periods_table.type);
% periods_table = addvars(periods_table, period_label_output' ,'NewVariableNames','type_label');
periods_table.type_label = period_label_output;

[period_label_output] = subfnBuildUpdatedTable(periods_table.behavioral_epoch);
% periods_table = addvars(periods_table, period_label_output' ,'NewVariableNames','behavioral_epoch_label');
periods_table.behavioral_epoch_label = period_label_output;

num_period_entries = height(periods_table);
periods_table.period_label = cellstr(string([1:num_period_entries]))';

if table_was_loaded_from_workspace
    % If the table was loaded from the workspace variable, save the updated table to the workspace variable as well.
%     assignin('base', periods_table_workspace_variable_name, periods_table);
%     assignin('caller', periods_table_workspace_variable_name, periods_table);
end

    function [period_label_output] = subfnBuildUpdatedTable(period_entries)
        [unique_entries, ~, ~] = unique(period_entries);
        num_unique_entries = length(unique_entries);
        accumulator_count = ones([num_unique_entries 1],"int64");
        num_period_entries = length(period_entries);
        period_label_output = cell([num_period_entries 1]);
        for i = 1:num_period_entries
            curr_entry = period_entries(i);
            curr_type_count = accumulator_count(curr_entry);
            period_label_output{i} = sprintf('%s_%d', curr_entry, curr_type_count);
            % Update the counter for this variable
            accumulator_count(curr_entry) = accumulator_count(curr_entry) + 1;
        end
    end

end