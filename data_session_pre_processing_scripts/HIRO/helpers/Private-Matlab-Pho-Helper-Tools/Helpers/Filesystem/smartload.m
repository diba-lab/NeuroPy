function [varargout] = smartload(varargin)
	% smartload.m: aims to be a drop-in replacement for the builtin MATLAB load command that doesn't repeatitively load in the same huge object from .mat file if it already exists in the workspace.
    
    is_load_skipped = false;

	if ~exist('special_options','var')
		special_options = struct; 
	end
	special_options = fnAddDefaultOptionalArgs({'display_loading_messages', 'force_overwrite'}, ...
				{true, false}, ...
				special_options);

    matFilePath = varargin{1};

    if ~exist(matFilePath,"file")
        is_load_skipped = false;
        error(sprintf('smartload !! Error: the provided file "%s" does not exist!', matFilePath));
    else
        
        %% Check user-provided variables to know whether a load needs to occur
        if nargin > 1
            provided_variable_names = varargin(2:end); % the user provided variable names to load specifically, check if they already exist in the default workspace
            found_flag_indicies = [];
            %% Loop through the provided arguments and make sure they aren't flags
            for aPotentialNonVariableIndex = 1:length(provided_variable_names)
                curr_var_name = provided_variable_names{aPotentialNonVariableIndex};
                if ~isvarname(curr_var_name)
                    % If the passed variable name isn't a valid one, check and see if it's a flag
                    switch curr_var_name
                        case {'-force','-f'}
                            % Force-overwrite
                            special_options.force_overwrite = true;
                            found_flag_indicies(end+1) = aPotentialNonVariableIndex; % and the flag index to remove it (since it's invalid)
                        otherwise
                           fprintf('Warning, variable named %s is not a valid variable name, nor is it a known flag!\n', curr_var_name);
                    end % end switch
                end % end if ~isvarname
            end % end for aPotentialNonVariableIndex
            % remove the invalid indicies
%             provided_variable_names{found_flag_indicies} = [];
             provided_variable_names(found_flag_indicies) = [];
             varargin(found_flag_indicies+1) = []; % is this safe?
        else
            [provided_variable_names] = fnGetMatFileVariableNames(matFilePath); % Use all variables in the mat file if none are specified
            
            if ~isempty(provided_variable_names)
                [flat_suggested_variable_list_string] = fnPrintCellStringOfVariableNames(provided_variable_names);    
            end
        end % end if
    
        %% Check for the variables in the caller workspace and see if they already exist
        if ~isempty(provided_variable_names)
            for i = 1:length(provided_variable_names)
                does_exist(i) = evalin('caller', sprintf('exist(''%s'', ''var'')', provided_variable_names{i}));    
            end
        
            if (sum(does_exist) == length(provided_variable_names))
                fprintf('smartload - All variables already exist in the base workspace.');
                if ~special_options.force_overwrite
                    is_load_skipped = true;
                    fprintf('The file will not be loaded\n');
                else
                    warning(sprintf('special_options.force_overwrite is true, so the file will be loaded overwritting the workspace variables!\n'));
                end
            else
                if exist('flat_suggested_variable_list_string','var')
                    % Print the found variables so the user can update the function call if they desire
                    fprintf('found variables: %s\n', flat_suggested_variable_list_string);
                    fprintf('future calls can use: \n \t smartload(''%s'', ...\n\t%s);\n', matFilePath, flat_suggested_variable_list_string);
                end
            end
    
        else
           fprintf('smartload - Specified file contains no variables! The file will not be loaded\n');
           is_load_skipped = true;
        end

    end

    if ~is_load_skipped
        % Note that NARGOUT is the number of output
        % arguments you called the function with.
        
        if special_options.display_loading_messages
            fprintf('smartload(...): loading data from %s... Please wait...\n', matFilePath);
        end

        if nargout == 0
            %% No output parameters provided, set in base workspace
            % assignin("caller", curr_out_struct)
            load(varargin{:});
            if ~isempty(provided_variable_names)
                save_to_base(provided_variable_names); % call it with the provided variable names
            else
                save_to_base; % Call it with no arguments, which will trigger intractive mode
            end
        else
            % Output arguments provided
            curr_out_struct = load(varargin{:});
            [loaded_variable_names] = getMemberNames(curr_out_struct);
            num_loaded_variables = length(loaded_variable_names);
            if num_loaded_variables == nargout
                % If there's one output argument for each loaded variable, assign them one-to-one
                for ii = 1:nargout  
                    varargout{ii} = curr_out_struct.(loaded_variable_names{ii});
                end
    
            elseif nargout == 1
                % If only one output variable was provided, interpret it as a struct and return the entire loaded struct
                varargout{1} = curr_out_struct;
            else
                error('smartload !! Error: the provided output variables to assign to must match the variables loaded from the input array');
            end
        end

        if special_options.display_loading_messages
            fprintf('\t done.\n');
        end

    end % end if ~is_load_skipped


end