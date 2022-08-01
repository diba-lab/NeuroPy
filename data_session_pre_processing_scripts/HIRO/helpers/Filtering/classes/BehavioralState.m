classdef BehavioralState
    %BEHAVIORALSTATE Summary of this class goes here
    %   Detailed explanation goes here
    
   properties (Constant)
      classColors = [0.5, 0.5, 1.0
               0.7, 0.7, 1.0
               1.0, 0.7, 0.7
               1.0, 0.0, 0.0];
      classNames = {'nrem', 'rem', 'quiet', 'active'};
      classValues = [1:length(BehavioralState.classNames)];
    end % end Constant properties block

    methods
        function obj = BehavioralState()
            %BEHAVIORALSTATE Construct an instance of this class
            %   Detailed explanation goes here
        end
    end

    

    methods (Static)
        function [out] = convert(string_representation)
            % convert: builds a proper categorical output array from a cell array of strings corresponding to a name in Self.classNames
            % string_representation: a cell array of strings matching one of the names in classNames. Like, processing_config.active_expt.behavior_list(:,3)
            out = categorical(string_representation, BehavioralState.classValues, BehavioralState.classNames);
        end

        function [is_state_included] = filter(filter_target, included_states)
            % filter: filters the filter_target (such as a table column containing state information) to only include the states included in included_states
            %% Example: [is_state_included] = BehavioralState.filter(active_processing.behavioral_periods_table.type, {'rem'});

           num_of_target_entries = length(filter_target);
            if exist('included_states','var') & ~isempty(included_states)
                is_state_included = false([num_of_target_entries 1]);
                curr_conditions.is_behavioral_state_type = (@(compare_type) (filter_target == compare_type));
                temp.curr_is_included = cellfun(curr_conditions.is_behavioral_state_type, included_states, 'UniformOutput', false);
                % Iterate through all the conditions:
                for i = 1:length(temp.curr_is_included)
        %             is_period_included = is_period_included & temp.curr_is_included{i};
                    is_state_included = is_state_included | temp.curr_is_included{i}; % Should be or (|) as it must meet one of the requirements for epoch to be included.
                end
            else
                % When empty, return true for all of this type
                is_state_included = true([num_of_target_entries 1]);
            end
        end


        function [app] = buildListBoxGui(app, position, callbackFunction)
            % Create SleepStateListBox
            app.SleepStateListBox = uilistbox(app.UIFigure);
            app.SleepStateListBox.Items = {'Active Wake', 'Quiet Wake', 'NREM', 'REM'};
            app.SleepStateListBox.Multiselect = 'on';
            app.SleepStateListBox.ValueChangedFcn = createCallbackFcn(app, callbackFunction, true);
            app.SleepStateListBox.Position = position;
            app.SleepStateListBox.Value = {'Active Wake'};

        end


    end % end static method block

end

