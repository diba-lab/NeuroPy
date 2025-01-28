% Requires the datastructures from "PhoDibaPrepare_Stage0.m" to be loaded
% Stage 1 of the processing pipeline

% addpath(genpath('helpers'));
% addpath(genpath('libraries/buzcode/'));

clear temp

if ~exist('data_config','var')
    Config;
end

if ~exist('active_processing','var') %TEMP: cache the loaded data to rapidly prototype the script
    fprintf('loading from %s...\n', data_config.output.intermediate_file_paths{2});
    load(data_config.output.intermediate_file_paths{2}, 'active_processing', 'processing_config', 'num_of_electrodes', 'source_data', 'timesteps_array');
    fprintf('done.\n');
else
    fprintf('active_processing already exists in workspace. Using extant data.\n');
end


%% Pairwise Indexing:
% Generates all unique pairs of indicies for pairwise comparisons (without replacement or repetition)
general_results.indicies.unique_electrode_pairs = nchoose2([1:num_of_electrodes]);
general_results.indicies.num_unique_pairs = length(general_results.indicies.unique_electrode_pairs);

% Build a reverse lookup matrix
general_results.indicies.reverse_lookup_unique_electrode_pairs = zeros(num_of_electrodes);
for linear_pair_idx = 1:general_results.indicies.num_unique_pairs
	curr_pair = general_results.indicies.unique_electrode_pairs(linear_pair_idx,:);
	general_results.indicies.reverse_lookup_unique_electrode_pairs(curr_pair(1), curr_pair(2)) = linear_pair_idx;
	general_results.indicies.reverse_lookup_unique_electrode_pairs(curr_pair(2), curr_pair(1)) = linear_pair_idx;
end


%% Get the duration of the states 
general_results.GroupedByState.groups = findgroups(active_processing.behavioral_periods_table.type);
general_results.GroupedByState.durations = splitapply(@sum, active_processing.behavioral_periods_table.duration, general_results.GroupedByState.groups);

if ~exist('OPTION_SKIP_STEP_SIZE_DEPENDENT_OUTPUTS','var')
    OPTION_SKIP_STEP_SIZE_DEPENDENT_OUTPUTS = false; % FALSE by default
end


if ~OPTION_SKIP_STEP_SIZE_DEPENDENT_OUTPUTS
    %% Build the Correlational Results:
    processing_config.max_xcorr_lag = 9; % Specified the maximum pairwise cross-correlation lag in seconds, the output ranges from -maxlag to maxlag
    num_of_behavioral_state_periods = height(active_processing.behavioral_periods_table);
    results_array = cell([length(processing_config.step_sizes) 1]);
    
    
    %% Loop through each listed binning/step size and built an "results_array"
    for current_binning_index = 1:length(processing_config.step_sizes)
	    active_binning_resolution = processing_config.step_sizes{current_binning_index};
	    temp.curr_timestamps = timesteps_array{current_binning_index};
	    temp.curr_processed = active_processing.processed_array{current_binning_index};
        temp.curr_step_size = processing_config.step_sizes{current_binning_index};
        
	    % Binned Spike rates per time:
	    % [temp.active_binned_spike_data_matrix] = sparse(fnUnitDataCells2mat(temp.curr_processed.all.binned_spike_counts));  % 35351x126 double
	    [temp.active_binned_spike_data_matrix] = fnUnitDataCells2mat(temp.curr_processed.all.binned_spike_counts);  % 35351x126 double
    
	    % Compute the correlational metrics for all data:
	    [~, active_results.all.autocorrelations, active_results.all.partial_autocorrelations, active_results.all.pairwise_xcorrelations] = fnProcessCorrelationalMeasures(temp.curr_processed.all.binned_spike_counts, general_results.indicies, processing_config, temp.curr_step_size);
    
	    %% Aggregate Measures:
    
	    %%% Split based on experiment epoch:
	    for i = 1:length(active_processing.definitions.behavioral_epoch.classNames)
		    temp.curr_epoch_name = active_processing.definitions.behavioral_epoch.classNames{i};
		    % size(temp.curr_processed.by_epoch.(temp.curr_epoch_name).binned_spike_counts): 1 x 126
		    [~, active_results.by_epoch.(temp.curr_epoch_name).autocorrelations, active_results.by_epoch.(temp.curr_epoch_name).partial_autocorrelations, active_results.by_epoch.(temp.curr_epoch_name).pairwise_xcorrelations] = fnProcessCorrelationalMeasures(temp.curr_processed.by_epoch.(temp.curr_epoch_name).binned_spike_counts, ...
			    general_results.indicies, processing_config, temp.curr_step_size);
    
		    active_results.aggregates.by_epoch.(temp.curr_epoch_name).spikes = cell2mat(temp.curr_processed.by_epoch.(temp.curr_epoch_name).binned_spike_counts);
		    active_results.aggregates.by_epoch.(temp.curr_epoch_name).across_all_cells.count = sum(active_results.aggregates.by_epoch.(temp.curr_epoch_name).spikes, 2);
		    active_results.aggregates.by_epoch.(temp.curr_epoch_name).total_counts = sum(active_results.aggregates.by_epoch.(temp.curr_epoch_name).spikes, 'all');
		    
		    fprintf('epoch: %s\n total_counts: %d\n', temp.curr_epoch_name, active_results.aggregates.by_epoch.(temp.curr_epoch_name).total_counts);
	    end
    
    
	    %timesteps
    
	    if processing_config.show_graphics
		    figure(1);
	    end
    
	    %%% Split based on behavioral state:
	    for i = 1:length(active_processing.definitions.behavioral_state.classNames)
		    temp.curr_state_name =  active_processing.definitions.behavioral_state.classNames{i};
		    
		    [~, active_results.by_state.(temp.curr_state_name).autocorrelations, active_results.by_state.(temp.curr_state_name).partial_autocorrelations, active_results.by_state.(temp.curr_state_name).pairwise_xcorrelations] = fnProcessCorrelationalMeasures(temp.curr_processed.by_state.(temp.curr_state_name).binned_spike_counts, ...
			    general_results.indicies, processing_config, temp.curr_step_size);
    
		    active_results.aggregates.by_state.(temp.curr_state_name).spikes = cell2mat(temp.curr_processed.by_state.(temp.curr_state_name).binned_spike_counts);
		    active_results.aggregates.by_state.(temp.curr_state_name).across_all_cells.count = sum(active_results.aggregates.by_state.(temp.curr_state_name).spikes, 2);
		    active_results.aggregates.by_state.(temp.curr_state_name).total_counts = sum(active_results.aggregates.by_state.(temp.curr_state_name).spikes, 'all');
		    
		    fprintf('state: %s\n total_counts: %d\n', temp.curr_state_name, active_results.aggregates.by_state.(temp.curr_state_name).total_counts);
		    
		    if processing_config.show_graphics
			    subplot(4,1,i);
			    plot(active_results.aggregates.by_state.(temp.curr_state_name).across_all_cells.count);
			    ylabel(temp.curr_state_name);
			    xlabel('');
		    end
	    end
    
    
	    if processing_config.show_graphics
		    xlim([timesteps(1), timesteps(end)]);
		    title('behavioral state spike counts')
	    end
    
    
	    %%% Split based on behavioral period:
	    active_results.all.timestamp_to_behavioral_period_map = zeros(size(temp.curr_timestamps));
       
        pairwise_xcorrelations.lag_offsets = (-processing_config.max_xcorr_lag):active_binning_resolution:processing_config.max_xcorr_lag;
        pairwise_xcorrelations.num_lag_steps = length(pairwise_xcorrelations.lag_offsets);
        %% max_xcorr_lag must be specified in terms of samples (num unit timesteps), not seconds, so we must convert by dividing by the currStepSize
        max_xcorr_lag_unit_time = processing_config.max_xcorr_lag / active_binning_resolution;
    
        % Pre-allocate output matricies:
        
        % active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_full: [num_of_behavioral_state_periods x num_unique_pairs x num_lag_steps] array
        active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_full = zeros([num_of_behavioral_state_periods, general_results.indicies.num_unique_pairs, pairwise_xcorrelations.num_lag_steps]);   
	    %     active_results.by_behavioral_period.pairwise_xcorrelations.xcorr = zeros([num_of_behavioral_state_periods, general_results.indicies.num_unique_pairs]);   
        
        % Loop over behavioral periods
        for behavioral_period_index = 1:num_of_behavioral_state_periods
            temp.curr_state_start = active_processing.behavioral_periods_table.epoch_start_seconds(behavioral_period_index);
            temp.curr_state_end = active_processing.behavioral_periods_table.epoch_end_seconds(behavioral_period_index);
            temp.curr_state_type = active_processing.behavioral_periods_table.type(behavioral_period_index);
            temp.curr_epoch_type = active_processing.behavioral_periods_table.behavioral_epoch(behavioral_period_index);
    
            % general_results.per_behavioral_state_period
            % active_processing.processed_array{1, 1}.by_state.rem.binned_spike_counts
    
            % TODO: validate these indices. Currently hacked up to avoid indexing errors.
            temp.curr_state_timesteps_start = ceil(temp.curr_state_start / active_binning_resolution) + 1;
            temp.curr_state_timesteps_end = min(floor(temp.curr_state_end / active_binning_resolution) + 1, (length(temp.curr_timestamps) - 1)); % constrain to valid index
    
            active_results.all.timestamp_to_behavioral_period_map(temp.curr_state_timesteps_start:temp.curr_state_timesteps_end) = behavioral_period_index;
        
            % Filter down to only the portion of the matrix that applies to the behavioral period.
        %     [~, ~, ~, active_results.by_behavioral_period.pairwise_xcorrelations.xcorr(behavioral_period_index, :)] = fnProcessCorrelationalMeasures(temp.active_binned_spike_data_matrix(temp.curr_state_timesteps_start:temp.curr_state_timesteps_end, :), ...
        %         general_results.indicies, processing_config, active_binning_resolution);
    
    %         temp.output_pairwise_xcorrelations = zeros([general_results.indicies.num_unique_pairs pairwise_xcorrelations.num_lag_steps]); % 7875x181 double
            for j = 1:general_results.indicies.num_unique_pairs
               active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_full(behavioral_period_index, j, :) = xcorr(temp.active_binned_spike_data_matrix(temp.curr_state_timesteps_start:temp.curr_state_timesteps_end, general_results.indicies.unique_electrode_pairs(j,1)), ...
                   temp.active_binned_spike_data_matrix(temp.curr_state_timesteps_start:temp.curr_state_timesteps_end, general_results.indicies.unique_electrode_pairs(j,2)), ...
                   max_xcorr_lag_unit_time); % 181x1 double
            end
    
            % temp.output_pairwise_xcorrelations is 7875x19 (one for each timestep)
        %     [~, active_results.by_epoch.(temp.curr_epoch_name).autocorrelations, active_results.by_epoch.(temp.curr_epoch_name).partial_autocorrelations, active_results.by_epoch.(temp.curr_epoch_name).pairwise_xcorrelations] = fnProcessCorrelationalMeasures(temp.curr_processed.by_epoch.(temp.curr_epoch_name).binned_spike_counts, ...
        % 			general_results.indicies, processing_config, active_binning_resolution);
    
            fprintf('behavioral state progress: %d/%d\n', behavioral_period_index, num_of_behavioral_state_periods);
    
        end
    
        active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_all_lags = mean(active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_full, 3); % [num_of_behavioral_state_periods x num_unique_pairs] array
        active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_all_pairs = squeeze(mean(active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_full, 2)); % [num_of_behavioral_state_periods x num_lag_steps] array
        active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_all_periods = squeeze(mean(active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_full, 1)); % [num_unique_pairs x num_lag_steps] array
        
        %% Doubly Collapsed:
        active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_all_pairs_AND_lags = squeeze(mean(active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_all_pairs, 2)); % [num_of_behavioral_state_periods x 1]
    
	    %% Finally, assign the active_results to the results array
	    results_array{current_binning_index} = active_results;
	    
    end % end for processing_config.step_sizes loop
    
    if ~data_config.output.skip_saving_intermediate_results
        fprintf('writing out results (both general_results and results_array) to %s...\n', data_config.output.results_file_path);
        save(data_config.output.results_file_path, 'general_results', 'results_array');
    end
else
    %% Otherwise OPTION_SKIP_STEP_SIZE_DEPENDENT_OUTPUTS is True, and we only write out the 'general_results'
    if ~data_config.output.skip_saving_intermediate_results
        fprintf('writing out only general_results results to %s...\n', data_config.output.results_file_path);
        save(data_config.output.results_file_path, 'general_results');
    end
end % end for OPTION_SKIP_STEP_SIZE_DEPENDENT_OUTPUTS
fprintf('done.\n');

fprintf('PhoDibaProcess_Stage1 complete!\n');