% PhoNeuroPyConvert_ExportAllToPython_MAIN.m
% PhoNeuroPyConvert_ExportAllToPython_MAIN - Aims to export all of the
% .mat-format files needed to load Hiro and other data into Spike3D.
% data to NeuroPy 2022-07-08 Session format.
% Makes use of active_processing.position_table, active_processing.behavioral_epochs.start_seconds
% 
% Can use
% C:\Users\pho\repos\PhoDibaLab_REM_HiddenMarkovModel\PhoDibaProcess_ComputeSingleExperiment.m
% to load active_processing into memory, and then run this script directly
% instead of loading anything from disk.
%
% Author: Pho Hale
% PhoHale.com 
% email: halechr@umich.edu
% Created: 08-07-2022 ; Last revision: 08-07-2022 
% History: was based off of "PhoDibaConvert_SpikesToPython.m"

addpath(genpath('helpers'));
addpath(genpath('libraries/buzcode/'));


%%%%%%%%% CONFIG
% import_root_path = 'R:\data\RoyMaze1';
% import_file_name = 'PhoResults_Expt1_RoyMaze1.mat';


% export_root_path = '/Users/pho/repo/Python Projects/PhoNeuronGillespie2021CodeRepo/PhoMatlabDataScripting/ExportedData';
% export_root_path = 'R:\rMBP Python Repos 2022-07-07\PhoNeuronGillespie2021CodeRepo\PhoMatlabDataScripting\ExportedData';
export_root_path = 'R:\data\Hiro';

%% Filtering Options:
filter_config.filter_included_cell_types = {};
% filter_config.filter_maximum_included_contamination_level = {2};
filter_config.filter_maximum_included_contamination_level = {};
% filter_config.showOnlyAlwaysStableCells = true;
filter_config.showOnlyAlwaysStableCells = false;


% R:\data\RoyMaze1

% addpath(genpath('../../helpers'));
% addpath(genpath('../../libraries/buzcode/'));

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
%% Begin:
fprintf('PhoNeuroPyConvert_ExportAllToPython_MAIN ready to process!\n');


%% Experiment Options:
active_expt_index = 1;
if exist('active_experiment_names','var')
    active_experiment_name = active_experiment_names{active_expt_index};
else
    active_experiment_name = processing_config.active_expt.name; % get from processing config
end

%% Get filter info for active units
[filtered_outputs.filter_active_units, filtered_outputs.original_unit_index] = fnFilterUnitsWithCriteria(active_processing, filter_config.showOnlyAlwaysStableCells, filter_config.filter_included_cell_types, ...
    filter_config.filter_maximum_included_contamination_level);
temp.num_active_units = sum(filtered_outputs.filter_active_units, 'all');
fprintf('Filter: Including %d of %d total units\n', temp.num_active_units, length(filtered_outputs.filter_active_units));

%% Apply the filters:
PhoDibaTest_PositionalAnalysis_temp.active_spikes = active_processing.spikes.time(filtered_outputs.filter_active_units);
PhoDibaTest_PositionalAnalysis_temp.cell_indicies = filtered_outputs.original_unit_index; % Get the indicies of the remaining cells:
PhoDibaTest_PositionalAnalysis_temp.spike_cells = cellfun(@(cell_idx) [(cell_idx .* ones(size(PhoDibaTest_PositionalAnalysis_temp.active_spikes{cell_idx}'))), PhoDibaTest_PositionalAnalysis_temp.active_spikes{cell_idx}'], ...
    num2cell([1:temp.num_active_units]), ...
    'UniformOutput', false); %% This is the essential loading function: relies on .active_spikes


%% Pho Spikes and Spikes Cell Info 2022-07-08
PhoDibaTest_PositionalAnalysis_temp.speculated_unit_type = cellstr(active_processing.spikes.speculated_unit_type(filtered_outputs.filter_active_units));
% behavioral_epochs = [[0:(height(active_processing.behavioral_epochs)-1)]', table2array(active_processing.behavioral_epochs)];

%% Properties are ['shank', 'cluster', 'aclu', 'qclu']
PhoDibaTest_PositionalAnalysis_temp.shank = active_processing.spikes.id(filtered_outputs.filter_active_units, 1); % shank
PhoDibaTest_PositionalAnalysis_temp.cluster = active_processing.spikes.id(filtered_outputs.filter_active_units, 2); % cluster
PhoDibaTest_PositionalAnalysis_temp.aclu = PhoDibaTest_PositionalAnalysis_temp.cell_indicies; % (aclu)
PhoDibaTest_PositionalAnalysis_temp.qclu = active_processing.spikes.quality(filtered_outputs.filter_active_units); % quality (qclu)
PhoDibaTest_PositionalAnalysis_temp.speculated_unit_contamination_level = active_processing.spikes.speculated_unit_contamination_level(filtered_outputs.filter_active_units); % speculated_unit_contamination_level

% Make the path for the active experiment:
active_experiment_export_root_path = fullfile(export_root_path, active_experiment_name, 'ExportedData');
mkdir(active_experiment_export_root_path);

%% Spikes:
% The critical properties are: 'spike_cells'
fprintf('Saving spikes analysis data to %s...\n', fullfile(active_experiment_export_root_path, 'spikesAnalysis.mat'));
spike_cells_ids = PhoDibaTest_PositionalAnalysis_temp.cell_indicies;
spike_cells = PhoDibaTest_PositionalAnalysis_temp.spike_cells;
% spike_matrix = PhoDibaTest_PositionalAnalysis_temp.activeMatrix;
shank = PhoDibaTest_PositionalAnalysis_temp.shank;
cluster = PhoDibaTest_PositionalAnalysis_temp.cluster;
aclu = PhoDibaTest_PositionalAnalysis_temp.aclu;
qclu = PhoDibaTest_PositionalAnalysis_temp.qclu;
speculated_unit_contamination_level = PhoDibaTest_PositionalAnalysis_temp.speculated_unit_contamination_level;
speculated_unit_type = PhoDibaTest_PositionalAnalysis_temp.speculated_unit_type;
save(fullfile(active_experiment_export_root_path, 'spikesAnalysis.mat'), 'spike_cells', 'spike_cells_ids', 'shank', 'cluster', 'aclu', 'qclu', 'speculated_unit_contamination_level', 'speculated_unit_type','-v7.3')
fprintf('done!\n');
fprintf('Spikes export complete!\n');


%% Save out positionalAnalysis data for Python:
% Requires: active_processing.position_table and active_processing.speed_table

% Compute displacements per timestep, velocities per timestep, etc.
positionalAnalysis.displacement.dt = [diff(active_processing.position_table.timestamp)];
positionalAnalysis.displacement.dx = [diff(active_processing.position_table.x)];
positionalAnalysis.displacement.dy = [diff(active_processing.position_table.y)];
positionalAnalysis.displacement.speeds = sqrt(((positionalAnalysis.displacement.dx .^ 2)) + (positionalAnalysis.displacement.dy .^ 2)) ./ positionalAnalysis.displacement.dt;
% Add the initial zero (indicating no change on the initial point) to keep length
positionalAnalysis.displacement.dt = [0; diff(active_processing.position_table.timestamp)];
positionalAnalysis.displacement.dx = [0; diff(active_processing.position_table.x)];
positionalAnalysis.displacement.dy = [0; diff(active_processing.position_table.y)];
positionalAnalysis.displacement.speeds = [0; positionalAnalysis.displacement.speeds];
% Find the start and end times the animal was put on the track, which is the period in which positions are relevant.
positionalAnalysis.track_epoch.start_end_delay = 0.5; % positionalAnalysis.track_epoch.start_end_delay: a buffer at the start and end of the epoch to account for the recording not being setup quite yet.
positionalAnalysis.track_epoch.begin = active_processing.behavioral_epochs.start_seconds(2) + positionalAnalysis.track_epoch.start_end_delay;
positionalAnalysis.track_epoch.end = active_processing.behavioral_epochs.end_seconds(2) - positionalAnalysis.track_epoch.start_end_delay;
% Filter the timestamps where the animal was on the track:
positionalAnalysis.track_indicies = ((positionalAnalysis.track_epoch.begin < active_processing.position_table.timestamp) & (active_processing.position_table.timestamp < positionalAnalysis.track_epoch.end));
positionalAnalysis.track_position.t = active_processing.position_table.timestamp(positionalAnalysis.track_indicies);
positionalAnalysis.track_position.x = active_processing.position_table.x(positionalAnalysis.track_indicies);
positionalAnalysis.track_position.y = active_processing.position_table.y(positionalAnalysis.track_indicies);
positionalAnalysis.track_position.speeds = positionalAnalysis.displacement.speeds(positionalAnalysis.track_indicies);
% Filter the displacents in case we want those as well:
positionalAnalysis.displacement.dt = positionalAnalysis.displacement.dt(positionalAnalysis.track_indicies);
positionalAnalysis.displacement.dx = positionalAnalysis.displacement.dx(positionalAnalysis.track_indicies);
positionalAnalysis.displacement.dy = positionalAnalysis.displacement.dy(positionalAnalysis.track_indicies);
positionalAnalysis.track_position.xyv = [positionalAnalysis.track_position.x, positionalAnalysis.track_position.y, positionalAnalysis.track_position.speeds]; % a matrix with 3 columns corresponding to the x-pos, y-pos, and speed
% To have complete info, we need: positionalAnalysis.track_position.t, positionalAnalysis.track_position.xyv
% Compute the new bounds restricted to the valid (track) positions:
[positionalAnalysis.plotting.bounds.x(1), positionalAnalysis.plotting.bounds.x(2)] = bounds(positionalAnalysis.track_position.x);
[positionalAnalysis.plotting.bounds.y(1), positionalAnalysis.plotting.bounds.y(2)] = bounds(positionalAnalysis.track_position.y);
fprintf('Saving positional analysis data to %s...\n', fullfile(active_experiment_export_root_path, 'positionAnalysis.mat'));
save(fullfile(active_experiment_export_root_path, 'positionAnalysis.mat'), 'positionalAnalysis','-v7.3');
fprintf('done!\n');
fprintf('Positions export complete!\n');

%% Save out Extras for Python:
fprintf('Saving extras analysis data to %s...\n', fullfile(active_experiment_export_root_path, 'extrasAnalysis.mat'));
% Epoch names from table Row header:
behavioral_epoch_names = active_processing.behavioral_epochs.Row;
% Numerical table version:
behavioral_epochs = [[0:(height(active_processing.behavioral_epochs)-1)]', table2array(active_processing.behavioral_epochs)];
behavioral_periods = [[0:(height(active_processing.behavioral_periods_table)-1)]', double(active_processing.behavioral_periods_table.epoch_start_seconds), double(active_processing.behavioral_periods_table.epoch_end_seconds), double(active_processing.behavioral_periods_table.duration), double(active_processing.behavioral_periods_table.behavioral_epoch), double(active_processing.behavioral_periods_table.type)];
save(fullfile(active_experiment_export_root_path, 'extrasAnalysis.mat'), 'behavioral_epochs', 'behavioral_periods', 'behavioral_epoch_names','-v7.3')
fprintf('done!\n');
fprintf('Extras export complete!\n');






% ------------- END OF CODE --------------