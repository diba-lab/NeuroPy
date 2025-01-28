%% Config.m
% Specifies the configuration variables used by the pipeline scripts

% data_config.root_parent_path = '/Users/pho/Dropbox/Classes/Spring 2021/PIBS 600 - Rotations/Rotation_3_Kamran Diba Lab/DataProcessingProject/Hiro_Datasets';
% data_config.source_data_prefix = 'src';
% data_config.output_data_prefix = 'Results';

% data_config.root_parent_path = '/Volumes/iNeo/Data/Rotation_3_Kamran Diba Lab/ClusterFreeAnalysisProject/Data/Achilles_10252013';
data_config.root_parent_path = 'R:\data\Achilles_10252013'; % Windows path
data_config.source_data_prefix = 'src';
data_config.output_data_prefix = 'ExportedData';
data_config.output.intermediate_file_names = {'PhoIntermediate_Stage0_0.mat', 'PhoIntermediate_Stage0_1.mat'};

data_config.source_root_path = fullfile(data_config.root_parent_path, data_config.source_data_prefix);
data_config.output.root_path = fullfile(data_config.root_parent_path, data_config.output_data_prefix);
data_config.output.intermediate_file_paths = cellfun((@(filename) fullfile(data_config.output.root_path, filename)), data_config.output.intermediate_file_names, 'UniformOutput', false);
data_config.output.skip_saving_intermediate_results = false; % If True, intermediate results are saved to disk. Otherwise they're just kept in memory.

% microseconds (10^6): 1000000
% nanoseconds (10^9): 1000000000
data_config.conversion_factor = (10^6);
% active_processing.definitions.behavioral_epoch.classNames = {'pre_sleep', 'track', 'post_sleep'};
% active_processing.definitions.behavioral_epoch.classValues = [1:length(active_processing.definitions.behavioral_epoch.classNames)];


% Process one of the experiments: 
processing_config.active_expt.name = 'RoyMaze1';
processing_config.step_sizes = {0.1}; % Step Sizes in seconds 
processing_config.num_step_sizes = length(processing_config.step_sizes);
processing_config.max_xcorr_lag = 9; % Specified the maximum pairwise cross-correlation lag in seconds, the output ranges from -maxlag to maxlag

%% Results:
% Add the results file path:
data_config.output.skip_saving_final_results = false; % If true, final results are NOT saved to disk and are just kept in memory.
data_config.output.results_file_name = 'PhoResults.mat';
data_config.output.results_file_path = fullfile(data_config.output.root_path, data_config.output.results_file_name);




%% Configure Graphics and Plotting:
processing_config.show_graphics = false;
% Options for tightening up the subplots:
plotting_options.should_use_custom_subplots = false;

if plotting_options.should_use_custom_subplots
    plotting_options.subtightplot.gap = [0.01 0.01]; % [intra_graph_vertical_spacing, intra_graph_horizontal_spacing]
    plotting_options.subtightplot.width_h = [0.01 0.05]; % Looks like [padding_bottom, padding_top]
    plotting_options.subtightplot.width_w = [0.025 0.01];
    plotting_options.opt = {plotting_options.subtightplot.gap, plotting_options.subtightplot.width_h, plotting_options.subtightplot.width_w}; % {gap, width_h, width_w}
    subplot_cmd = @(m,n,p) subtightplot(m, n, p, plotting_options.opt{:});
else
    subplot_cmd = @(m,n,p) subplot(m, n, p);
end


% showOnlyAlwaysStableCells: shows only cells that are stable across all three behavioral epochs
processing_config.showOnlyAlwaysStableCells = true;
