function [ax, rectangle_handles] = fnPlotPeriodsDiagram(startTimes, endTimes, period_identity, plottingOptions)
%fnPlotPeriodsDiagram Plots a behavioral-state indicator as a function of time
%   Detailed explanation goes here

%% History:
% Generalized from fnPlotStateDiagram.m on 2021-10-26 by Pho Hale


%% Usage Example:
% periods_plotting_options.state_names = active_processing.definitions.behavioral_state.classNames;
% periods_plotting_options.state_colors = [0.5, 0.5, 1.0
%                0.7, 0.7, 1.0
%                1.0, 0.7, 0.7
%                1.0, 0.0, 0.0];
% 
% [ax, rectangle_handles] = fnPlotPeriodsDiagram(active_processing.behavioral_periods_table.epoch_start_seconds, ...
%  	active_processing.behavioral_periods_table.epoch_end_seconds, ...
% 	active_processing.behavioral_periods_table.type, ...
% 	periods_plotting_options);

    if ~exist('plottingOptions','var')
         plottingOptions = struct();
    end
    
    plottingOptions = fnAddDefaultOptionalArgs({'orientation', 'vertical_state_mode', 'x_axis', 'state_names', 'state_colors', 'axis_tag'}, ...
        {'horizontal', 'combined', 'timestamp', {}, [], ''}, ...
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






	num_periods = length(startTimes);
	
	if strcmpi(plottingOptions.x_axis, 'timestamp')
		states = [startTimes, ...
			endTimes, ...
			double(period_identity)];
	else 
		linear_end_indicies = [1:num_periods];
		linear_start_indicies = linear_end_indicies - 1;
		states = [linear_start_indicies', ...
			linear_end_indicies', ...
			double(period_identity)];
	end
	
	%% TODO: Build default state_names/state_colors appropriately
	[unique_period_identities, ~, iC] = unique(period_identity);
	num_unique_period_identities = length(unique_period_identities);

	num_provided_state_names = length(plottingOptions.state_names);
	if (num_provided_state_names < num_unique_period_identities) 
		% Inadequate state names provided in plottingOptions, generate the rest
		num_state_names_to_generate = num_unique_period_identities - num_provided_state_names;
		for i = 1:num_state_names_to_generate
			plottingOptions.state_names{end+1} = sprintf('State_%d', (i + num_provided_state_names));
		end
	end
            
	num_provided_state_colors = length(plottingOptions.state_colors);
	if (num_provided_state_colors < num_unique_period_identities) 
		% Inadequate state colors provided in plottingOptions, generate the rest
		num_state_colors_to_generate = num_unique_period_identities - num_provided_state_colors;
		error('Not yet implemented! Please provide plottingOptions.state_colors with the same number of colors as the number of unique period identities provided in period_identity!')
		% for i = 1:num_state_colors_to_generate
		% 	plottingOptions.num_provided_state_colors{end+1} = sprintf('State_%d', (i + num_provided_state_names));
		% end
	end


    if isempty(plottingOptions.axis_tag)
        plottingOptions.axis_tag = sprintf('periodsDiagram_%d_uniqueIdentities', num_unique_period_identities);
    end

	% Coming in, we need:
	% 	states:
	% 	state_names:
	% 	color_state:

    state_names = plottingOptions.state_names;
    color_state = plottingOptions.state_colors;

	% states(end, 2) is the end time of the final period.
%     t = 0:1:states(end, 2); 

	rectangle_handles = cell([num_periods, 1]);
    
    for s_idx = 1:num_periods
        if strcmpi(plottingOptions.vertical_state_mode, 'stacked')
            curr_s_y = states(s_idx,3)-0.5;
        else
            % in combined mode they all have the same y-position.
            curr_s_y = 1.0-0.5;
        end
        
        % 
        if strcmpi(plottingOptions.orientation, 'vertical')
            % If vertically oriented, flip the x and y values
            temp.rect_pos = [curr_s_y, states(s_idx,1), 1, diff(states(s_idx,1:2))];
        else
            temp.rect_pos = [states(s_idx,1), curr_s_y, diff(states(s_idx,1:2)), 1];
        end
        
		%% Build the rectangle that will ultimately be plotted
        rectangle_handles{s_idx} = rectangle('Position', temp.rect_pos,...
                    'LineStyle','none', ...
					'facecolor', color_state(states(s_idx,3),:), ...
					'tag', state_names{states(s_idx,3)});
    end

	% Get the axes as ax
    ax=gca;
    ax.Tag = plottingOptions.axis_tag;

	%% Configure based on plottingOptions.vertical_state_mode:
    if strcmpi(plottingOptions.vertical_state_mode, 'stacked')
        state_stack_dim.Tick = 1:length(state_names);
        state_stack_dim.TickLabel = state_names;
        state_stack_dim.Lim = 0.5+[0,length(state_names)];
        
    else
        % combined mode: 
		% No ticks or TickLabels in combined mode
        state_stack_dim.Tick = [];
        state_stack_dim.TickLabel = '';
        state_stack_dim.Lim = 0.5 + [0,1];
    end
    
    state_stack_dim.Dir='reverse';
    
%     length_dim.Lim = t([1,end]);
%     length_dim.Lim = [0, states(end, 2)]; % For relative times, this should be more efficient

    % Try using the earliest and latest timestamp
    length_dim.Lim = [states(1, 1), states(end, 2)];

    length_dim.Axis.Visible = 'off';

	%% Configure based on plottingOptions.orientation:
     if strcmpi(plottingOptions.orientation, 'vertical')
        % If vertically oriented, flip the x and y values
        ax.XTick = state_stack_dim.Tick;
        ax.XTickLabel = state_stack_dim.TickLabel;
        ax.XLim = state_stack_dim.Lim;
        ax.XDir = state_stack_dim.Dir;
        
        ax.YLim = length_dim.Lim;
        ax.YAxis.Visible = length_dim.Axis.Visible;
        
     else
	 	% horizontally oriented
        ax.YTick = state_stack_dim.Tick;
        ax.YTickLabel = state_stack_dim.TickLabel;
        ax.YLim = state_stack_dim.Lim;
        ax.YDir = state_stack_dim.Dir;
        
        ax.XLim = length_dim.Lim;
        ax.XAxis.Visible = length_dim.Axis.Visible;
     end
    
end


