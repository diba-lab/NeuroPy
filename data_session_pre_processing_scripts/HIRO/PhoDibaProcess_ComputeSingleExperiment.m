clc
clear all; 
% addpath(genpath('helpers'));
% addpath(genpath('libraries/buzcode/'));


OPTION_SKIP_STEP_SIZE_DEPENDENT_OUTPUTS = true;

% clear all;

temp.possibleRootPaths = {'W:\Data\Rotation_3_Kamran Diba Lab\DataProcessingProject\Hiro_Datasets', ...
    '/Volumes/iNeo/Data/Rotation_3_Kamran Diba Lab/DataProcessingProject/Hiro_Datasets', ...
	'/Users/pho/Dropbox/Classes/Spring 2021/PIBS 600 - Rotations/Rotation_3_Kamran Diba Lab/DataProcessingProject/Hiro_Datasets'};

temp.possibleExperiments = {'RoyMaze1',	'RoyMaze2',	'RoyMaze3',	'TedMaze1',	'TedMaze2',	'TedMaze3',	'KevinMaze1'};
% active_expt_name = 'RoyMaze1';
% active_expt_name = 'RoyMaze2';
% active_expt_name = 'RoyMaze3';
% active_expt_name = 'TedMaze1';
% active_expt_name = 'TedMaze2';
% active_expt_name = 'TedMaze3';
active_expt_name = 'KevinMaze1';

% % auxillary pink rMBP:
% active_root_path = temp.possibleRootPaths{2};
% main rMBP:
% active_root_path = temp.possibleRootPaths{2};

% Windows Apogee:
active_root_path = temp.possibleRootPaths{1};

active_step_sizes = {10.0};
fprintf('fnRunPipeline starting for active_expt_name: %s...\n', active_expt_name);
[data_config, processing_config, plotting_options] = fnDataConfigForExptName(active_expt_name, active_step_sizes, active_root_path);
data_config.output.skip_saving_intermediate_results = true; % Skip saving intermediate results to disk
data_config.output.skip_saving_final_results = true; % Skip saving final results to disk as well

PhoDibaPrepare_Stage0 % active_processing, source_data, timesteps_array
PhoDibaProcess_Stage1 % general_results, results_array
PhoDibaProcess_Stage2 % general_results

fprintf('fnRunPipeline completed for active_expt_name: %s\n', active_expt_name);

PhoNeuroPyConvert_ExportAllToPython_MAIN

