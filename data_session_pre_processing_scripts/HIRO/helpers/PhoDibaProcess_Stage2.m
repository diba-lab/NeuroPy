% Requires the datastructures from "PhoDibaProcess_Stage1.m" to be loaded
% Stage 2 of the processing pipeline

% addpath(genpath('helpers'));
% addpath(genpath('libraries/buzcode/'));

clear temp

if ~exist('data_config','var')
    Config;
end


if ~exist('active_processing','var') %TEMP: cache the loaded data to rapidly prototype the script
    fprintf('loading data from %s...\n', data_config.output.intermediate_file_paths{2});
    load(data_config.output.intermediate_file_paths{2}, 'active_processing', 'processing_config', 'num_of_electrodes', 'source_data', 'timesteps_array');
    fprintf('done.\n');
else
    fprintf('active_processing already exists in workspace. Using extant data.\n');
end

% if ~exist('results_array','var') %TEMP: cache the loaded data to rapidly prototype the script
%     fprintf('loading results from %s...\n', data_config.output.results_file_path);
%     load(data_config.output.results_file_path, 'general_results', 'results_array');
%     fprintf('done. Contains results for %d different bin sizes.\n', length(results_array));
% else
%     fprintf('results_array already exists in workspace. Contains results for %d different bin sizes. Using extant data.\n', length(results_array));
% end


if ~exist('general_results','var') %TEMP: cache the loaded data to rapidly prototype the script
    fprintf('loading general_results from %s...\n', data_config.output.results_file_path);
    load(data_config.output.results_file_path, 'general_results');
    fprintf('done.\n');
else
    fprintf('general_results already exists in workspace.\n');
end

%% Begin:
fprintf('PhoDibaProcess_Stage2 ready to process!\n');

% Cluster ISIs:
% kmeanscluster(


% Get the duration of the epochs {'pre_sleep','track','post_sleep'}
% active_processing.behavioral_periods_table.behavioral_epochs.duration



% active_processing.spikes.behavioral_duration_indicies

%% Across all cells:
general_results.flattened_across_all_units.spike_timestamp = cat(2, active_processing.spikes.time{:});
general_results.flattened_across_all_units.spike_state_index = cat(1, active_processing.spikes.behavioral_duration_indicies{:});
general_results.flattened_across_all_units.spike_state = cat(1, active_processing.spikes.behavioral_states{:});
general_results.flattened_across_all_units.spike_epoch = cat(1, active_processing.spikes.behavioral_epoch{:});


%% For each behavioral state period, every unit fires a given number of spikes.
%	The average firing rate for each unit within that period is given by this number of spikes divided by the duration of that period.
    
num_of_behavioral_state_periods = height(active_processing.behavioral_periods_table);
general_results.edges = 1:num_of_behavioral_state_periods;

general_results.per_behavioral_state_period.num_spikes_per_unit = zeros([num_of_behavioral_state_periods num_of_electrodes]); %% num_results: 668x126 double
general_results.per_behavioral_state_period.spike_rate_per_unit = zeros([num_of_behavioral_state_periods num_of_electrodes]); %% num_results: 668x126 double


for unit_index = 1:num_of_electrodes

    % temp.edges = unique(active_processing.spikes.behavioral_duration_indicies{unit_index});
    general_results.counts = histc(active_processing.spikes.behavioral_duration_indicies{unit_index}(:), general_results.edges);
    general_results.per_behavioral_state_period.num_spikes_per_unit(:, unit_index) = general_results.counts;
    general_results.per_behavioral_state_period.spike_rate_per_unit(:, unit_index) = general_results.counts ./ active_processing.behavioral_periods_table.duration;
    % active_processing.spikes.behavioral_duration_indicies{unit_index}

    % behavioral_periods_table.duration

end

if ~data_config.output.skip_saving_intermediate_results
    fprintf('writing out general_results to %s...\n', data_config.output.results_file_path);
    save(data_config.output.results_file_path, 'general_results');
    fprintf('done.\n');
end
% 
fprintf('PhoDibaProcess_Stage2 complete!\n');



