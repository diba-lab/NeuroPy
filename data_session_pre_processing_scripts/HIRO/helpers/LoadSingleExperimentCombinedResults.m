%% LoadSingleExperimentCombinedResults.m
% Pho Hale, March 18, 2021
% Loads single experiment combined output files such as those produced by 'LoadFromMultiExperimentResults.m'.
% active_processing and results_array from the single experiment combined output file like 'PhoResults_Expt1_RoyMaze1.mat'.
% specified experiment index for compatibility with single-experiment scripts.

addpath(genpath('../helpers'));
addpath(genpath('../libraries/buzcode/'));

% User must specify the correct experiment name and its corresponding index (as they're used to build the filename to load):
% multi_experiment_config.active_experiment_id: the index of the active experiment to load
multi_experiment_config.active_experiment_id = 1;
multi_experiment_config.active_experiment_name = 'RoyMaze2';

if ~exist('data_config','var')
    Config;
end


% Loads a single experiment from a combined results file:
if ~exist('active_processing','var') 
    %% Set the path to the combined single experiment file:
    % data_config.output.all_expts_combined_parent_path = '/Users/pho/Dropbox/Classes/Spring 2020/PIBS 600 - Rotations/Rotation_3_Kamran Diba Lab/DataProcessingProject/Hiro_Datasets';
    % data_config.output.all_expts_combined_parent_path = '/Volumes/iNeo/Data/Rotation_3_Kamran Diba Lab/DataProcessingProject/Hiro_Datasets/Results';
    data_config.output.all_expts_combined_parent_path = 'F:\Data\Rotation_3_Kamran Diba Lab\DataProcessingProject\Hiro_Datasets\Results';
    data_config.output.active_expt_combined_results_file_name = sprintf('PhoResults_Expt%d_%s.mat', multi_experiment_config.active_experiment_id, multi_experiment_config.active_experiment_name);
    data_config.output.active_expt_combined_results_file_path = fullfile(data_config.output.all_expts_combined_parent_path, data_config.output.active_expt_combined_results_file_name);

    fprintf('\t loading combined single-experiment results from %s... (this will take quite a while, a few minutes or so...\n', data_config.output.active_expt_combined_results_file_path);
    temp.variablesList = {'active_processing', 'processing_config', 'num_of_electrodes', 'source_data', 'timesteps_array','results_array','general_results'};
    load(data_config.output.active_expt_combined_results_file_path, temp.variablesList{:});

else
    error('active_processing already exists. Refusing to override it.');
end

%% Add the multi-experiment structure for compatibility with existing multi-experiment scripts:
% Tries to build valid 'across_experiment_results', 'active_experiment_names', 'active_step_sizes' variables


if ~exist('across_experiment_results','var')
    %across_experiment_results'    
    for i = 1:length(temp.variablesList)
        across_experiment_results{multi_experiment_config.active_experiment_id}.(temp.variablesList{i}) = eval(temp.variablesList{i});
    end

end

if ~exist('active_experiment_names', 'var')
    active_experiment_names{multi_experiment_config.active_experiment_id} = multi_experiment_config.active_experiment_name;
end
    
% Rederive the active_step_sizes from the spacing (timestep) of the tempsteps_array
if ~exist('active_step_sizes', 'var')
    num_active_step_sizes = length(timesteps_array);
    
    active_step_sizes = zeros([num_active_step_sizes, 1]);
    
    for i = 1:num_active_step_sizes
        temp.stepSizes = diff(timesteps_array{i});
        active_step_sizes(i) = seconds(temp.stepSizes(1));
    end
    
end    


fprintf('done.\n');