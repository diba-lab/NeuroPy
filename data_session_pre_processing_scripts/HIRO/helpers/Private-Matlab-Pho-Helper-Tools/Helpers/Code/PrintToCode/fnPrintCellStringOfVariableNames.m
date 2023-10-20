function [variable_list_string] = fnPrintCellStringOfVariableNames(variable_names)
%     variable_names =
    %   7×1 cell array
    %     {'active_processing'}
    %     {'general_results'  }
    %     {'num_of_electrodes'}
    %     {'processing_config'}
    %     {'results_array'    }
    %     {'source_data'      }
    %     {'timesteps_array'  }

	% Output: 'active_processing', 'general_results', 'num_of_electrodes', 'processing_config', 'results_array', 'source_data', 'timesteps_array'

    variable_list_string = join(variable_names, ''', ''');
    variable_list_string = variable_list_string{1};
    
    % Add the appropriate start and end single quote:
    variable_list_string = ['''' variable_list_string ''''];
end