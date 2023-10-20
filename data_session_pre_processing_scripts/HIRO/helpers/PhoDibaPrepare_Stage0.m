% PhoDibaPrepare_Stage0
% Stage 0 of the processing pipeline.

% addpath(genpath('helpers'));
% addpath(genpath('libraries/buzcode/'));
% clear all;

if ~exist('data_config','var')
    Config;
end

if ~exist('active_processing','var') %TEMP: cache the loaded data to rapidly prototype the script
    [active_processing, source_data] = loadData(data_config, processing_config);
end

%% Preprocessing:

% Each entry in active_processing.spikes has a variable number of double entries, indicating the relative offset (in seconds) the spike occured for each unit.
num_of_electrodes = height(active_processing.spikes);

%% For each behavioral period in behavioral_periods_table:
% we want to be able to extract:
%% Any spikes that occur within that period
%% the experimental_phase it belongs in {pre_sleep, track, post_sleep}

%%
%%%%% spikes table pre-processing:

%% Find units that are stable across all sessions:
active_processing.spikes.stability_count = sum(active_processing.spikes.isStable, 2);
active_processing.spikes.isAlwaysStable = (active_processing.spikes.stability_count == 3);
numAlwaysStableCells = sum(active_processing.spikes.isAlwaysStable, 'all');


%% Compute ISIs:
active_processing.spikes.ISIs = cellfun((@(spikes_timestamps) diff(spikes_timestamps)), ...
 active_processing.spikes.time, 'UniformOutput', false);

active_processing.spikes.meanISI = cellfun((@(spikes_ISIs) mean(spikes_ISIs)), ...
 active_processing.spikes.ISIs, 'UniformOutput', false);

active_processing.spikes.ISIVariance = cellfun((@(spikes_ISIs) var(spikes_ISIs)), ...
 active_processing.spikes.ISIs, 'UniformOutput', false);


%% Partition spikes based on behavioral state:
temp.curr_num_of_behavioral_states = height(active_processing.behavioral_periods_table);
temp.spikes_behavioral_states = cell([num_of_electrodes, 1]);
temp.spikes_behavioral_epoch = cell([num_of_electrodes, 1]);

% Pre-allocation: Loop over electrodes
for electrode_index = 1:num_of_electrodes
    % Convert spike times to relative to expt start and scale to seconds. 
    temp.spikes_behavioral_states{electrode_index} = categorical(ones([active_processing.spikes.num_spikes(electrode_index), 1]), active_processing.definitions.behavioral_state.classValues, active_processing.definitions.behavioral_state.classNames);
    temp.spikes_behavioral_epoch{electrode_index} = categorical(ones([active_processing.spikes.num_spikes(electrode_index), 1]), [1:length(active_processing.definitions.behavioral_epoch.classNames)], active_processing.definitions.behavioral_epoch.classNames);
    active_processing.spikes.behavioral_duration_indicies{electrode_index} = zeros([active_processing.spikes.num_spikes(electrode_index), 1]); % to store the index of the corresponding behavioral state the spike belongs to
end


% Loop over behavioral activities
for state_index = 1:temp.curr_num_of_behavioral_states
    temp.curr_state_start = active_processing.behavioral_periods_table.epoch_start_seconds(state_index);
    temp.curr_state_end = active_processing.behavioral_periods_table.epoch_end_seconds(state_index);
    temp.curr_state_type = active_processing.behavioral_periods_table.type(state_index);
    temp.curr_epoch_type = active_processing.behavioral_periods_table.behavioral_epoch(state_index);
    
    fprintf('behavioral state progress: %d/%d\n', state_index, temp.curr_num_of_behavioral_states);
    
    temp.curr_state_spikes = cell(num_of_electrodes, 1);
    % Extract the spike train for each electrode
    for electrode_index = 1:num_of_electrodes
        % Convert spike times to relative to expt start and scale to seconds.
        temp.curr_electrode_spikes = active_processing.spikes.time{electrode_index};
        % Get the spike times that belong to this particular state.
        temp.curr_state_spikes_idx{electrode_index} = find((temp.curr_state_start < temp.curr_electrode_spikes) & (temp.curr_electrode_spikes < temp.curr_state_end));
        
        temp.curr_state_spikes{electrode_index} = temp.curr_electrode_spikes((temp.curr_state_start < temp.curr_electrode_spikes) & (temp.curr_electrode_spikes < temp.curr_state_end));
        
        temp.spikes_behavioral_states{electrode_index}(temp.curr_state_spikes_idx{electrode_index}) = temp.curr_state_type;
        temp.spikes_behavioral_epoch{electrode_index}(temp.curr_state_spikes_idx{electrode_index}) = temp.curr_epoch_type;
        
        active_processing.spikes.behavioral_duration_indicies{electrode_index}(temp.curr_state_spikes_idx{electrode_index}) = state_index;
    end

end

active_processing.spikes.behavioral_states = temp.spikes_behavioral_states;
active_processing.spikes.behavioral_epoch = temp.spikes_behavioral_epoch;

if ~data_config.output.skip_saving_intermediate_results
    fprintf('writing out to %s...\n', data_config.output.intermediate_file_paths{1});
    save(data_config.output.intermediate_file_paths{1}, 'active_processing', 'data_config', 'processing_config', 'num_of_electrodes', 'source_data');
    fprintf('done.\n');
end

% Smooth with a gaussian window
% Perform Gaussian Binning: (sigma = 1.5 sec)

%% TODO: this is where refinements would be made
timesteps_array = cellfun((@(dt) seconds(active_processing.behavioral_epochs.start_seconds(1):dt:active_processing.behavioral_epochs.end_seconds(end))), ...
 processing_config.step_sizes, 'UniformOutput', false);

active_processing.processed_array = cell([processing_config.num_step_sizes 1]);

for i = 1:length(processing_config.step_sizes)
	% timesteps_array{i} = seconds(active_processing.behavioral_epochs.start_seconds(1):processing_config.step_sizes{i}:active_processing.behavioral_epochs.end_seconds(end));
	[active_processing.processed_array{i}] = fnPreProcessSpikeData(active_processing, data_config, num_of_electrodes, timesteps_array{i});
	% [active_processing.processed_array{i}] = fnBinSpikeData(active_processing, data_config, timesteps_array{i});
end

if ~data_config.output.skip_saving_intermediate_results
    fprintf('writing out to %s...\n', data_config.output.intermediate_file_paths{2});
    save(data_config.output.intermediate_file_paths{2}, 'active_processing', 'data_config', 'processing_config', 'num_of_electrodes', 'source_data', 'timesteps_array');
    fprintf('done.\n');
end

fprintf('PhoDibaPrepare_Stage0 complete!\n');





