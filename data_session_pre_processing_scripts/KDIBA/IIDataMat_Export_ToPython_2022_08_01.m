%% IIDataMat_Export_ToPython_2022_08_01.m
%% Used to process Kamran's oldest recording session format, such as those located:
%       https://drive.google.com/drive/folders/1NmVVTFCqgxO0tzis9b4hTbVfkVmTvPDw
%% Loads a IIdata.mat file such as the one in 'R:\data\KDIBA\gor01\one\2006-6-08_14-26-15\..\IIdata.mat'

%% Required Format Files:
% Parent directory combined data file: '../IIdata.mat':
% Session directory spike file: 'spike_file = r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53\2006-6-07_11-26-53.spikeII.mat'
% Neuroscope XML file: neuroscope_xml_file = Path(basedir).joinpath('2006-6-07_11-26-53.xml')'
%
addpath(genpath('PhoHelpers'));


%% Outputs:
% Outputs: '*.position_info.mat', '*.epochs_info.mat', '*.laps_info.mat', '*.spikes.mat'

%% Config:
parent_dir = 'R:\data\KDIBA\gor01\one';
session_name = '2006-6-07_11-26-53';
% session_name = '2006-6-08_14-26-15';

%% Common:
session_folder = fullfile(parent_dir, session_name);
session_export_path = session_folder;
microseconds_to_seconds_conversion_factor = 1/1e6;


%% IIdata.mat:
%% Exports two .mat files in the session directory:
% R:\data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.position_info.mat
%   Containing recorded behavioral position information.
% R:\data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.epochs_info.mat:
%   Containing epoch/start/stop info.
load(fullfile(parent_dir,'IIdata.mat'), 'IIdata');
IIdataTable = struct2table(IIdata,'AsArray',true);
active_idx = find(strcmpi(IIdataTable.name, session_name));
active_IIdata = IIdata(active_idx, :); % active_IIdata: get the data for only this session

position.timestamps = active_IIdata.xyt2(:,3) .* microseconds_to_seconds_conversion_factor; % convert microseconds to milliseconds?
position.position.x = active_IIdata.xyt2(:,1) .* active_IIdata.pix2cm;
position.position.y = active_IIdata.xyt2(:,2) .* active_IIdata.pix2cm;

% compute sampling rate:
position_sampling_rate_Hz = 1.0 ./ mean(diff(position.timestamps)); % In Hz
position.samplingRate = position_sampling_rate_Hz;
%[active_IIdata.tbegin, active_IIdata.tend]
%active_IIdata.chts % the change time
epoch_data = [active_IIdata.tbegin, active_IIdata.chts;
    active_IIdata.chts, active_IIdata.tend];
epoch_data = epoch_data .* microseconds_to_seconds_conversion_factor; % convert to seconds
epoch_data_rel = epoch_data - epoch_data(1,1);

%% File Saving:
samplingRate = position.samplingRate;
timestamps = position.timestamps;
timestamps_rel = timestamps - (active_IIdata.tbegin * microseconds_to_seconds_conversion_factor);
x = position.position.x;
y = position.position.y;
save(fullfile(session_export_path, [session_name, '.position_info.mat']), 'position', ...
    'microseconds_to_seconds_conversion_factor','samplingRate', 'timestamps', 'timestamps_rel', 'x', 'y', ...
    '-v7.3');
save(fullfile(session_export_path, [session_name, '.epochs_info.mat']), 'epoch_data','epoch_data_rel','microseconds_to_seconds_conversion_factor', '-v7.3');
% Both epochs and positions are saved in units of seconds and specified in
% absolute times

%% .spikeII.mat:
datSampleRate = 32552.1; % Assumed to be in Hz, "Dat sampling rate"
[~, spike, laps_info, was_changed] = PhoPrepareSpikesOutput(parent_dir, session_name, active_IIdata.tbegin, datSampleRate);

