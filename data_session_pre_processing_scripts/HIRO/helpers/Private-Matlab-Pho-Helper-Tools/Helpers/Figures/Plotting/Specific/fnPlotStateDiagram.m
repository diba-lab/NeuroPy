function [ax, rectangle_handles] = fnPlotStateDiagram(active_processing, plottingOptions)
%fnPlotStateDiagram Plots a behavioral-state indicator as a function of time
%   Detailed explanation goes here

    if ~exist('plottingOptions','var')
         plottingOptions = struct();
    end
    
    plottingOptions = fnAddDefaultOptionalArgs({'orientation', 'vertical_state_mode', 'plot_variable', 'x_axis', 'axis_tag'}, ...
        {'vertical', 'combined', 'behavioral_state', 'timestamp', ''}, ...
        plottingOptions);
    
%         plottingOptions.orientation = 'vertical';
% %         plottingOptions.orientation = 'horizontal';
%         
%     %     plottingOptions.vertical_state_mode = 'stacked';
%         plottingOptions.vertical_state_mode = 'combined';
%         
%         plottingOptions.plot_variable = 'behavioral_epoch';
%     %     plottingOptions.plot_variable = 'behavioral_state';
%         
%         plottingOptions.x_axis = 'timestamp'; % Timestamp-appropriate relative bins
% %         plottingOptions.x_axis = 'index'; % Equal-sized bins
%         

    if isempty(plottingOptions.axis_tag)
        plottingOptions.axis_tag = sprintf('stateDiagram_%s', plottingOptions.plot_variable);
    end

    if strcmpi(plottingOptions.plot_variable, 'behavioral_epoch')
        % behavioral_epoch
        if ~isfield(active_processing.definitions.behavioral_epoch, 'classColors')
            active_processing.definitions.behavioral_epoch.classColors = [0.0, 0.5, 0.0
               0.2, 1.0, 0.2
               0.0, 0.2, 0.0];
        end
     
        startTimes = active_processing.behavioral_periods_table.epoch_start_seconds;
        endTimes = active_processing.behavioral_periods_table.epoch_end_seconds;
        period_identity = active_processing.behavioral_periods_table.behavioral_epoch;

        plottingOptions.state_names = active_processing.definitions.behavioral_epoch.classNames;
        plottingOptions.state_colors = active_processing.definitions.behavioral_epoch.classColors;

    elseif strcmpi(plottingOptions.plot_variable, 'behavioral_state')
        % behavioral_state
        if ~isfield(active_processing.definitions.behavioral_state, 'classColors')
            active_processing.definitions.behavioral_state.classColors = [0.5, 0.5, 1.0
               0.7, 0.7, 1.0
               1.0, 0.7, 0.7
               1.0, 0.0, 0.0];
        end
        
        startTimes = active_processing.behavioral_periods_table.epoch_start_seconds;
        endTimes = active_processing.behavioral_periods_table.epoch_end_seconds;
        period_identity = active_processing.behavioral_periods_table.type;

        plottingOptions.state_names = active_processing.definitions.behavioral_state.classNames;
        plottingOptions.state_colors = active_processing.definitions.behavioral_state.classColors;

    else
        error('invalid plottingOptions.plot_variable');
    end
     

    %t = 0:1:states(end, 2);

    [ax, rectangle_handles] = fnPlotPeriodsDiagram(startTimes, endTimes, period_identity, plottingOptions);

end

