classdef BehavioralEpoch
    %BEHAVIORALEPOCH Summary of this class goes here
    %   Detailed explanation goes here
  
   properties (Constant)
      classColors = [0.0, 0.5, 0.0
               0.2, 1.0, 0.2
               0.0, 0.2, 0.0];
      classNames = {'pre_sleep', 'track', 'post_sleep'}; % shouldn't this be 'pre', 'track', 'post'?
      classValues = [1:length(BehavioralEpoch.classNames)];
    end % end Constant properties block

    methods
        function obj = BehavioralEpoch()
            %BehavioralEpoch Construct an instance of this class
            %   Detailed explanation goes here
        end
    end




       
   methods (Static)
       function [is_epoch_included] = filter(filter_target, included_epochs)
           % filter: filters the filter_target (such as a table column containing epoch type information) to only include the epoch types included in included_epochs
           %% Example: [is_epoch_included] = BehavioralEpoch.filter(active_processing.behavioral_periods_table.behavioral_epoch, {'pre_sleep','track'});
           num_of_target_entries = length(filter_target);
            if exist('included_epochs','var') & ~isempty(included_epochs)
                is_epoch_included = false([num_of_target_entries 1]);
                curr_conditions.is_behavioral_epoch_type = (@(compare_type) (filter_target == compare_type));
                temp.curr_is_included = cellfun(curr_conditions.is_behavioral_epoch_type, included_epochs, 'UniformOutput', false);
                % Iterate through all the conditions:
                for i = 1:length(temp.curr_is_included)
        %             is_period_included = is_period_included & temp.curr_is_included{i};
                    is_epoch_included = is_epoch_included | temp.curr_is_included{i}; % Should be or (|) as it must meet one of the requirements for epoch to be included.
                end
            else
                % When empty, return true for all of this type
                is_epoch_included = true([num_of_target_entries 1]);
            end
       end

    end % end static method block

end

