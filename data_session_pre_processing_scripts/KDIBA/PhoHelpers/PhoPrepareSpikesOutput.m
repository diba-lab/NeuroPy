function [out_paths, spike, laps_info, was_changed] = PhoPrepareSpikesOutput(parent_dir, session_name, tbegin_uSec, SampleRateHz, enable_seconds_primary_output)
%PHOPREPARESPIKESOUTPUT Summary of this function goes here
%   Outputs: '*.laps_info.mat', '*.spikes.mat'
%
% e.g. parent_dir = 'R:\data\KDIBA\gor01\one';
%   session_name = '2006-6-07_11-26-53';
    session_folder = fullfile(parent_dir, session_name);
    session_export_path = session_folder;
    %load(fullfile(parent_dir,'IIdata.mat'), 'IIdata');
    S = load(fullfile(session_folder, [session_name '.spikeII.mat']), 'spike');
    spike = S.spike;
    % 5902: -1
    % 5903: 1
    % 11309: 1
    % 11310: -1
    % 19997: -1
    % 19998: 2
    
    if ~exist('enable_seconds_primary_output','var')
        enable_seconds_primary_output = false; %if true, returns start_t and end_t in seconds format as the primary format.
    end
    
    % Conversion:
    Spktimes_microseconds = tbegin_uSec + ((spike.t * 1e6)/SampleRateHz); % known to be in units of microseconds
    Spktimes_seconds = Spktimes_microseconds / 1e6; % This converts the spike.t column into absoulte seconds
    % Add the 't_seconds' column:
    spike.t_seconds = Spktimes_seconds;
    spike.t_rel_seconds = Spktimes_seconds - (tbegin_uSec / 1e6);
    spike.t_res = spike.t;

    if enable_seconds_primary_output
        spike.t = spike.t_seconds; % write out the t_seconds as the primary t variable
    end

    [updated_lap_ids, updated_maze_ids, laps_info, was_changed] = PhoPrepareLapsOutput(spike.lap);
    if was_changed
        %spike.t_seconds = spike.t .* spikes_t_to_seconds_conversion_factor;
        spike.maze_relative_lap = spike.lap;
        spike.lap = updated_lap_ids;
        spike.maze_id = updated_maze_ids;
    
        % save out the laps info:
        lap_id = laps_info.ids;    
        maze_id = laps_info.maze_id;
        start_spike_index = laps_info.start_indicies;
        end_spike_index = laps_info.end_indicies;

        % In absolute seconds time coordinates:
        laps_info.start_t_seconds = spike.t_seconds(laps_info.start_indicies);
        laps_info.end_t_seconds = spike.t_seconds(laps_info.end_indicies);
        laps_info.duration_seconds = laps_info.end_t_seconds - laps_info.start_t_seconds;

        % In .res time coordinates:
        laps_info.start_t_res = spike.t_res(laps_info.start_indicies);
        laps_info.end_t_res = spike.t_res(laps_info.end_indicies);

        % laps_info.start_t and end_t will automatically take the primary
        % output time format from spike.t
        laps_info.start_t = spike.t(laps_info.start_indicies);
        laps_info.end_t = spike.t(laps_info.end_indicies);

        laps_info.duration = laps_info.end_t - laps_info.start_t;

        start_t = laps_info.start_t;
        end_t = laps_info.end_t;
        start_t_seconds = laps_info.start_t_seconds;
        end_t_seconds = laps_info.end_t_seconds;
        duration_seconds = laps_info.duration_seconds;
    
        out_paths = {fullfile(session_export_path, [session_name, '.laps_info.mat']), ...
            fullfile(session_export_path, [session_name, '.spikes.mat']) ...
        };
        save(fullfile(session_export_path, [session_name, '.laps_info.mat']), 'lap_id', ...
            'maze_id','start_spike_index', 'end_spike_index', 'start_t', 'end_t', 'start_t_seconds', 'end_t_seconds', 'duration_seconds', ...
            '-v7.3');

%         save(fullfile(session_export_path, [session_name, '.laps_info.mat']), 'lap_id', ...
%             'maze_id','start_spike_index', 'end_spike_index', 'start_t', 'end_t', 'start_t_seconds', 'end_t_seconds', 'duration_seconds', ...
%             '-v7.3');
        save(fullfile(session_export_path, [session_name, '.spikes.mat']), 'spike', '-v7.3');
        
    end

end

