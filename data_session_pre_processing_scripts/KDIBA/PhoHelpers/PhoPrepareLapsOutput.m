function [spike_lap_ids, spike_maze_id, laps_info, was_changed] = PhoPrepareLapsOutput(spike_lap_ids)
    % Ensure monotonically increasing lap_ids

    spike_maze_id = ones(size(spike_lap_ids));
    neg_one_indicies = find(spike_lap_ids == -1);

%     temp_indicies = [1:length(lap_ids)];
    temp_diff = [diff(spike_lap_ids); 0];
    change_indices = find(temp_diff ~= 0) + 1; % 160x1
%     change_diff_values = temp_diff(change_indices);
    change_values = spike_lap_ids(change_indices);
    
    rising_edge_indicies = change_indices(change_values ~= -1);
    falling_edge_indicies = [change_indices(change_values == -1); length(spike_lap_ids)];
    
    start_id_indicies = find(change_values == 1);
%     start_id_falling_edge_indicies = start_id_indicies + 1;
    
    if length(start_id_indicies) > 1
        % If there are non-monotonic duplicates (more than one lap starting
        % at 1 with other laps in between.
        second_start_id_idx = start_id_indicies(2);
        split_index = change_indices(second_start_id_idx); % ans = 616656 % the location that starts the first repeat
        % change_indices(start_id_falling_edge_indicies(2)) % 624241
        
        % get the lap_id of the last lap in the pre-split
        pre_split_lap_idx = change_indices(second_start_id_idx - 2); % skip back two place to get the previous rising edge
        max_pre_split_lap_id = spike_lap_ids(pre_split_lap_idx);
        
        spike_maze_id(1:split_index-1) = 1;
        spike_maze_id(split_index:end) = 2;
        
        spike_lap_ids(split_index:end) = spike_lap_ids(split_index:end) + max_pre_split_lap_id; % adding the last pre_split lap ID means that the first lap starts at max_pre_split_lap_id + 1, the second max_pre_split_lap_id + 2, etc 
        spike_lap_ids(neg_one_indicies) = -1; % re-set any negative 1 indicies from the beginning back to negative 1

        was_changed = true;
        
    else
        % No non-monotonic duplicates, just return
        was_changed = false;
    end

    unique_lap_ids = unique(spike_lap_ids(spike_lap_ids ~= -1)); % do not return the -1 index
    % build an output laps_info object:
    laps_info.ids = unique_lap_ids;    
    laps_info.maze_id = ones(size(unique_lap_ids));
    laps_info.maze_id(laps_info.ids > max_pre_split_lap_id) = 2;

    laps_info.start_indicies = rising_edge_indicies;
    laps_info.end_indicies = falling_edge_indicies;

%     [rising_edge_indicies, falling_edge_indicies] % 80 x 2

end