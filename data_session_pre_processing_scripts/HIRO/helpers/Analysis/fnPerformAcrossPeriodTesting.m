

function [plotResults, filtered, results] = fnPerformAcrossPeriodTesting(active_processing, general_results, active_results, processing_config, filter_config, expt_info, plottingOptions)
    %%% fnPerformAcrossREMTesting: Run main analysis
    %%%
 
    plotResults.exports = {};
    plotResults.configs = {};
    
    %% Get filter info for active units
    [filter_config.filter_active_units, filter_config.filter_active_unit_original_indicies] = fnFilterUnitsWithCriteria(active_processing, processing_config.showOnlyAlwaysStableCells, filter_config.filter_included_cell_types, ...
        filter_config.filter_maximum_included_contamination_level);
    fprintf('Filter: Including %d of %d total units\n', sum(filter_config.filter_active_units, 'all'), length(filter_config.filter_active_units));
    
    temp.curr_pairs_indicies = find(filter_config.filter_active_units);
    filter_config.filter_active_pairs =  ismember(general_results.indicies.unique_electrode_pairs(:,1), temp.curr_pairs_indicies) & ismember(general_results.indicies.unique_electrode_pairs(:,2), temp.curr_pairs_indicies);
    filter_config.filter_active_pair_values = general_results.indicies.unique_electrode_pairs(filter_config.filter_active_pairs, :);
    
    
    %% Compute fields for all groups:
    for i = 1:plottingOptions.num_groups
        temp.curr_group_name = plottingOptions.group_name_list{i};
        
        %% Group Indexing:
        [filtered.(plottingOptions.group_indexArray_variableName_list{i})] = fnFilterPeriodsWithCriteria(active_processing, plottingOptions.group_included_epochs{i}, plottingOptions.group_included_states{i}); % 668x1
        temp.curr_group_indicies = filtered.(plottingOptions.group_indexArray_variableName_list{i});
        
        % Computations:
        results.(temp.curr_group_name).num_behavioral_periods = sum(temp.curr_group_indicies,'all');
        results.(temp.curr_group_name).per_period.durations = active_processing.behavioral_periods_table.duration(temp.curr_group_indicies);
        results.(temp.curr_group_name).per_period.epoch_start_seconds = active_processing.behavioral_periods_table.epoch_start_seconds(temp.curr_group_indicies);
        results.(temp.curr_group_name).per_period.epoch_end_seconds = active_processing.behavioral_periods_table.epoch_end_seconds(temp.curr_group_indicies);

        % Compute the center of the epochs to plot the firing rates along an appropriately scaled x-axis:
        results.(temp.curr_group_name).per_period.epoch_center_seconds = (results.(temp.curr_group_name).per_period.epoch_start_seconds + floor(results.(temp.curr_group_name).per_period.durations ./ 2.0));

         % Leave in terms of the spike rates per unit (14x92 double):
        results.(temp.curr_group_name).spike_rate_per_unit = general_results.per_behavioral_state_period.spike_rate_per_unit(temp.curr_group_indicies, filter_config.filter_active_units);

        % Average Across all of the units
        results.(temp.curr_group_name).spike_rate_all_units.mean = mean(results.(temp.curr_group_name).spike_rate_per_unit, 2); % 14x1 double
        results.(temp.curr_group_name).spike_rate_all_units.stdDev = std(results.(temp.curr_group_name).spike_rate_per_unit, 0, 2); % 14x1 double

        % Compute the average across the REM sessions in each epoch
        results.(temp.curr_group_name).baseline_spike_rate_across_all.mean = mean(results.(temp.curr_group_name).spike_rate_all_units.mean);
        results.(temp.curr_group_name).baseline_spike_rate_across_all.stdDev = std(results.(temp.curr_group_name).spike_rate_all_units.mean);
    
        %% xcorr_all_lags: averaged over lags, preserving all pairs.
        results.(temp.curr_group_name).per_period.xcorr_all_lags = active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_all_lags(temp.curr_group_indicies, filter_config.filter_active_pairs); % 668x7875
        
        %% xcorr_all_pairs:
        results.(temp.curr_group_name).per_period.xcorr_all_pairs = active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_all_pairs(temp.curr_group_indicies, :); % 668x81
        %% xcorr_all_pairs_AND_lags:
        results.(temp.curr_group_name).per_period.xcorr_all_pairs_AND_lags = active_results.by_behavioral_period.pairwise_xcorrelations.xcorr_all_pairs_AND_lags(temp.curr_group_indicies, :); % 668x1

        fprintf('%s: %d periods\n', temp.curr_group_name, results.(temp.curr_group_name).num_behavioral_periods);
        
    end
    
%     fprintf('any_REM: %d periods\n pre_sleep_REM: %d periods\n post_sleep_REM: %d periods\n', results.any_REM.num_behavioral_periods, results.pre_sleep_REM.num_behavioral_periods, results.post_sleep_REM.num_behavioral_periods);

    %%% Plotting Results:
    temp.curr_expt_string = sprintf('experiment[%d]: %s', expt_info.index, expt_info.name);
    plottingOptions.outputs.curr_expt_filename_string = sprintf('%s_', expt_info.name);
    
    
    %% TODO: Plot all others (states that aren't REM at all) as a separate series

    %% Plot Mean Firing Rates across units FOR ALL PERIODS:

    plotResults.figures.meanSpikeRateAllPeriodsFig = figure(119+expt_info.index);
    clf
    hold off;
    
    for i = 1:plottingOptions.num_groups
        
        if plottingOptions.group_plot_types.meanfiringrate{i} 
            temp.curr_group_name = plottingOptions.group_name_list{i};
            temp.curr_group_indicies = filtered.(plottingOptions.group_indexArray_variableName_list{i});

            if strcmpi(plottingOptions.plottingXAxis, 'index')
                plottingOptions.x = [1:results.(temp.curr_group_name).num_behavioral_periods];
            else
                plottingOptions.x = results.(temp.curr_group_name).per_period.epoch_center_seconds;
            end


            [h0] = fnPlotAcrossREMTesting(plottingOptions.plottingMode, plottingOptions.x, ...
                results.(temp.curr_group_name).spike_rate_all_units.mean, ...
                results.(temp.curr_group_name).spike_rate_all_units.stdDev, ...
                results.(temp.curr_group_name).spike_rate_per_unit);

            if ~strcmpi(plottingOptions.plottingMode, 'distributionPlot')
                %     h0(1).Marker = '*';
                h0(1).DisplayName = temp.curr_group_name;
            end

            hold on;
        
        end % end if plottingOptions.group_plot_types.meanfiringrate

    end
    
    
    % Set up axis properties:
    title(sprintf('Firing Rate All periods: %d', results.(temp.curr_group_name).num_behavioral_periods));
    if strcmpi(plottingOptions.plottingXAxis, 'index')
        xlabel('Filtered Period Index')
    else
        xlabel('Period Timestamp Offset (Seconds)')
    end
    ylabel('mean spike rate')
    if ~isempty(plottingOptions.plottingYlim)
        ylim(plottingOptions.plottingYlim)
    end

    legend();
    sgtitle([temp.curr_expt_string ' : Spike Rates - All Periods - Period Index - All Cells'])
    
    
    
    
%     
%     %% Plot Mean Firing Rates across units FOR REM PERIODS ONLY:
%     % Error bars are across units:
%     plotResults.figures.meanSpikeRateFig = figure(9+expt_info.index);
%     clf
%     subplot(2,1,1);
% 
%     if strcmpi(plottingOptions.plottingXAxis, 'index')
%         plottingOptions.x = [1:results.pre_sleep_REM.num_behavioral_periods];
%     else
%         plottingOptions.x = results.pre_sleep_REM.per_period.epoch_center_seconds;
%     end
% 
%     [h1] = fnPlotAcrossREMTesting(plottingOptions.plottingMode, plottingOptions.x, ...
%         results.pre_sleep_REM.spike_rate_all_units.mean, ...
%         results.pre_sleep_REM.spike_rate_all_units.stdDev, ...
%         results.pre_sleep_REM.spike_rate_per_unit);
% 
%     title(sprintf('PRE sleep REM periods: %d', results.pre_sleep_REM.num_behavioral_periods));
%     if strcmpi(plottingOptions.plottingXAxis, 'index')
%         xlabel('Filtered Period Index')
%     else
%         xlabel('Period Timestamp Offset (Seconds)')
%     end
%     ylabel('mean spike rate')
%     if ~isempty(plottingOptions.plottingYlim)
%         ylim(plottingOptions.plottingYlim)
%     end
% 
%     subplot(2,1,2);
% 
%     if strcmpi(plottingOptions.plottingXAxis, 'index')
%         plottingOptions.x = [1:results.post_sleep_REM.num_behavioral_periods];
%     else
%         plottingOptions.x = results.post_sleep_REM.per_period.epoch_center_seconds;
%     end
%     [h2] = fnPlotAcrossREMTesting(plottingOptions.plottingMode, plottingOptions.x, ...
%         results.post_sleep_REM.spike_rate_all_units.mean, ...
%         results.post_sleep_REM.spike_rate_all_units.stdDev, ...
%         results.post_sleep_REM.spike_rate_per_unit);
% 
%     title(sprintf('POST sleep REM periods: %d', results.post_sleep_REM.num_behavioral_periods));
%     if strcmpi(plottingOptions.plottingXAxis, 'index')
%         currPlotConfig.xlabel = 'Filtered Period Index';
%         
%     else
%         currPlotConfig.xlabel = 'Period Timestamp Offset (Seconds)';
% 
%     end
%     currPlotConfig.ylabel = 'mean spike rate';
%     
%     xlabel(currPlotConfig.xlabel)
%     ylabel(currPlotConfig.ylabel)
%     if ~isempty(plottingOptions.plottingYlim)
%         ylim(plottingOptions.plottingYlim)
%     end
%     sgtitle([temp.curr_expt_string ' : Spike Rates - PRE vs Post Sleep REM Periods - Period Index - Pyramidal Only'])
%     % Figure Name:
%     %'Spike Rates - PRE vs Post Sleep REM Periods - Period Index';
%     %'Spike Rates - PRE vs Post Sleep REM Periods - Timestamp Offset';
% 
% %     'Spike Rates - PRE vs Post Sleep REM Periods - Period Index - Pyramidal Only'
%     % MainPlotVariableBeingCompared - PurposeOfComparison - IndependentVariable - FiltersAndConstraints
%     
%     % Build Figure Export File path:
%     currPlotConfig.curr_expt_filename_string = sprintf('%s - %s - %s - %s - %s', ...
%         plottingOptions.outputs.curr_expt_filename_string, ...
%         'Spike Rates', ...
%         'PRE vs Post Sleep REM Periods', ...
%         currPlotConfig.xlabel, ...
%         'Spike Rates'...
%         );
% 
%     currPlotConfig.curr_expt_parentPath = plottingOptions.outputs.rootPath;
%     currPlotConfig.curr_expt_path = fullfile(currPlotConfig.curr_expt_parentPath, currPlotConfig.curr_expt_filename_string);
% 
%     % Perform the export:
%     [plotResults.exports{end+1}.export_result] = fnSaveFigureForExport(plotResults.figures.meanSpikeRateFig, currPlotConfig.curr_expt_path, true, false, false, true);
%     plotResults.configs{end+1} = currPlotConfig;
    
    %% Display the Correlational Results:
    plottingOptions.curr_expt_string = temp.curr_expt_string;
    
    %% Do for all indexes:
    temp.curr_units_to_test = filter_config.filter_active_unit_original_indicies;

%     temp.curr_units_to_test = filter_config.filter_active_unit_original_indicies(1:4);
    [temp.is_pair_included, temp.original_pair_index] = fnFilterPairsWithCriteria(general_results, temp.curr_units_to_test);
    % original_pair_index: 126x86 double
    
    filtered.is_pair_included = temp.is_pair_included(filter_config.filter_active_pairs, :); % 3655x86 logical
    filtered.original_pair_index = temp.original_pair_index(filter_config.filter_active_units, :); % 86x86 double
    
        % Sanity Check:
%     sum(temp.is_pair_included, 1);
%     sum(filtered.is_pair_included, 1);
    
   
    for i = 1:length(temp.curr_units_to_test)
        temp.curr_reference_unit_index = temp.curr_units_to_test(i);
        plotResults.figures.xcorr = fnPlotCellPairsByPeriodHeatmap(active_processing, general_results, results, filter_config, plottingOptions, temp.curr_reference_unit_index);
        
        % Build Figure Export File path:
        currPlotConfig.curr_expt_filename_string = sprintf('%s - %s - %s - %s - %s', ...
            plottingOptions.outputs.curr_expt_filename_string, ...
            'XCorr for all lags', ...
            'All Periods', ...
            'cellPairs', ...
            sprintf('unit[%d]', temp.curr_reference_unit_index));
    
        currPlotConfig.curr_expt_parentPath = fullfile(plottingOptions.outputs.rootPath, 'png');
        if ~exist(currPlotConfig.curr_expt_parentPath, 'dir')
           mkdir(currPlotConfig.curr_expt_parentPath); 
        end
        currPlotConfig.curr_expt_path = fullfile(currPlotConfig.curr_expt_parentPath, currPlotConfig.curr_expt_filename_string);

        % Perform the export:
        [plotResults.exports{end+1}.export_result] = fnSaveFigureForExport(plotResults.figures.xcorr, currPlotConfig.curr_expt_path, false, false, false, true);
        plotResults.configs{end+1} = currPlotConfig;

    end
    
end
