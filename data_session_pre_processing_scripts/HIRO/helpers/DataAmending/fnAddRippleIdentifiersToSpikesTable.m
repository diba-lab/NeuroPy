function [spikesTable] = fnAddRippleIdentifiersToSpikesTable(spikesTable, rippleStartTimes, rippleEndTimes)
% fnAddRippleIdentifiersToSpikesTable: From a table containing the spike data for each unit and two vectors of ripple event start/end times, determines whether each spike occurs during a ripple, and if so, what ripple index it belongs to. It adds these two columns to the existing spikesTable. 
% TODO: This is a suboptimal/inefficient implementation

%% Usage Example:
% ripples_time_mat = source_data.ripple.RoyMaze1.time;
% ripples_time_mat = (ripples_time_mat - source_data.behavior.RoyMaze1.time(1,1)) ./ 1e6; % Convert to relative timestamps since start
% startTimes = ripples_time_mat(:,1);
% endTimes = ripples_time_mat(:,2);
% [active_processing.spikes] = fnAddRippleIdentifiersToSpikesTable(active_processing.spikes, startTimes, endTimes);
% % Save changes back out to the file we loaded from:
% % save('/Volumes/iNeo/Data/Rotation_3_Kamran Diba Lab/DataProcessingProject/Hiro_Datasets/Results/PhoResults_Expt1_RoyMaze1.mat','active_processing','-append')


%% Inputs:
% spikesTable: should be a table in the form of active_processing.spikes

%% Example Input Table:

%% Updated Output Table: 


%% Testing: 
[outputFilterFunction] = fnBuildPeriodDetectingFilter(rippleStartTimes, rippleEndTimes);

% Pre-allocate the cell arrays that will be used as teh table columns
is_ripple_spike = cell([height(spikesTable) 1]);
spike_ripple_index = cell([height(spikesTable) 1]);

for unit_idx = 1:height(spikesTable)
    unit_matches = cellfun(@(period_comparison_fcn) find(period_comparison_fcn(spikesTable.time{unit_idx})~=0, 1, 'first'), functionCells, 'UniformOutput', false);
    unit_matched_periods = find(~cellfun(@isempty, unit_matches)); % These are the indicies of the periods that each spike falls within
    unit_matched_spike_indicies = [unit_matches{unit_matched_periods}]'; % These are the indicies of spikes for this unit that fall within a period
    
    % Finally we can go through and build the required row entry for each of the new columns for the spikes table:
    unit_is_ripple_spike = false(size(spikesTable.time{unit_idx}));
    unit_is_ripple_spike(unit_matched_spike_indicies) = true;
    % Set the accumulator column:
    is_ripple_spike{unit_idx} = unit_is_ripple_spike;

    unit_spike_ripple_indicies = NaN(size(spikesTable.time{unit_idx})); % Most spikes fall outside a period, and have a NaN value for the matched period index
    unit_spike_ripple_indicies(unit_matched_spike_indicies) = unit_matched_periods; % For the spikes that did match a period, set their value to the matched period index
    spike_ripple_index{unit_idx} = unit_spike_ripple_indicies;

end

% Finally, add the new columns to the table
spikesTable.isRippleSpike = is_ripple_spike;
spikesTable.RippleIndex = spike_ripple_index;


end