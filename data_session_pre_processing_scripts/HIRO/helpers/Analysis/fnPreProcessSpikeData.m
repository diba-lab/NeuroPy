function [output] = fnPreProcessSpikeData(active_processing, data_config, num_of_electrodes, timesteps)
%fnPreProcessSpikeData Summary of this function goes here
%   Detailed explanation goes here

    output_cell_size = [1, num_of_electrodes];
    
	
    
    
    %% Pre-allocation of output variables:
	output.all.spike_data = cell(output_cell_size);
    output.all.binned_spike_counts = cell(output_cell_size);
    %% Split based on experiment epoch:
    for i = 1:length(active_processing.definitions.behavioral_epoch.classNames)
        temp.curr_epoch_name = active_processing.definitions.behavioral_epoch.classNames{i};
        output.by_epoch.(temp.curr_epoch_name).spike_data = cell(output_cell_size);
        output.by_epoch.(temp.curr_epoch_name).binned_spike_counts = cell(output_cell_size);
    end

    %% Split based on behavioral state:
    for i = 1:length(active_processing.definitions.behavioral_state.classNames)
        temp.curr_state_name =  active_processing.definitions.behavioral_state.classNames{i};
        output.by_state.(temp.curr_state_name).spike_data = cell(output_cell_size);
        output.by_state.(temp.curr_state_name).binned_spike_counts = cell(output_cell_size);
    end
        
        
        
    %% Processing:    
	for electrode_index = 1:num_of_electrodes
		% Convert spike times to relative to expt start and scale to seconds. 
		fprintf('fnPreProcessSpikeData: electrode progress: %d/%d\n', electrode_index, num_of_electrodes);
		
		temp.curr_timetable = timetable(seconds(active_processing.spikes.time{electrode_index}'), active_processing.spikes.behavioral_epoch{electrode_index}, active_processing.spikes.behavioral_states{electrode_index}, active_processing.spikes.behavioral_duration_indicies{electrode_index}, ...
			'VariableNames',{'behavioral_epoch','behavioral_state','behavioral_period_index'});
		
		output.all.spike_data{electrode_index} = temp.curr_timetable;
		output.all.binned_spike_counts{electrode_index} = histcounts(output.all.spike_data{electrode_index}.Time, timesteps)';

		%% Split based on experiment epoch:
		for i = 1:length(active_processing.definitions.behavioral_epoch.classNames)
			temp.curr_epoch_name = active_processing.definitions.behavioral_epoch.classNames{i};
			output.by_epoch.(temp.curr_epoch_name).spike_data{electrode_index} = temp.curr_timetable((temp.curr_timetable.behavioral_epoch == temp.curr_epoch_name), :);
			output.by_epoch.(temp.curr_epoch_name).binned_spike_counts{electrode_index} = histcounts(output.by_epoch.(temp.curr_epoch_name).spike_data{electrode_index}.Time, timesteps)';
		end
		
		%% Split based on behavioral state:
		for i = 1:length(active_processing.definitions.behavioral_state.classNames)
			temp.curr_state_name =  active_processing.definitions.behavioral_state.classNames{i};
			output.by_state.(temp.curr_state_name).spike_data{electrode_index} = temp.curr_timetable((temp.curr_timetable.behavioral_state == temp.curr_state_name), :);
			output.by_state.(temp.curr_state_name).binned_spike_counts{electrode_index} = histcounts(output.by_state.(temp.curr_state_name).spike_data{electrode_index}.Time, timesteps)';
		end

	end % end for


	% output.all.binned_spike_counts = cell([num_of_electrodes, 1]);
	% for electrode_index = 1:num_of_electrodes
	%     % Gaussian smoothing of spike data:
	%     fprintf('progress: %d/%d\n', electrode_index, num_of_electrodes);
	%     % This retimed version is pretty slow: > 30 seconds execution time.
	%     output.normalized_spike_data{electrode_index} = retime(active_processing.processed.all.spike_data{electrode_index},'regular','count','TimeStep', seconds(1));    
	%     output.all.binned_spike_counts{electrode_index} = smoothdata(output.normalized_spike_data{electrode_index},'gaussian', seconds(1.5));
	% 
	%     histcounts(active_processing.processed.all.spike_data{electrode_index}, timesteps);
	%     
	%     
	%     %% Split based on experiment epoch:
	%     for i = 1:length(active_processing.definitions.behavioral_epoch.classNames)
	%         temp.curr_epoch_name = active_processing.definitions.behavioral_epoch.classNames{i};
	%         output.by_epoch.(temp.curr_epoch_name).spike_data{electrode_index} = temp.curr_timetable((temp.curr_timetable.behavioral_epoch == temp.curr_epoch_name), :);
	%     end
	%     
	%     %% Split based on behavioral state:
	%     for i = 1:length(active_processing.definitions.behavioral_state.classNames)
	%         temp.curr_state_name =  active_processing.definitions.behavioral_state.classNames{i};
	%         output.by_state.(temp.curr_state_name).spike_data{electrode_index} = temp.curr_timetable((temp.curr_timetable.behavioral_state == temp.curr_state_name), :);
	%     end
	%     
	%     
	% end

end

