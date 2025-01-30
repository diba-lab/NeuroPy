function save_to_base(local_variable_names, remote_variable_names)
% Save current workspace variables (such as the workspace created within a function call) to the base workspace

    is_interactive = false;

    if ~exist('local_variable_names','var')
        local_variable_names = {};
        is_interactive = true;
    end
    if ~exist('remote_variable_names','var')
        remote_variable_names = local_variable_names; % use the same names as the local names by default
    end

    vars_name = evalin('caller','who');
    if is_interactive
        [selection, ok] = listdlg('liststring',vars_name,'selectionmode','multiple',...
            'name','Select variable(s) in current workspace','listsize',[300,300]);
    else
        %% Non-interactive mode:
        selection = []; % empty selection to start
        for i = 1:length(local_variable_names)
            [~, idx] = ismember(local_variable_names{i}, vars_name);
%             idx = find(strcmp([vars_name{:}], local_variable_names{i}), 1); % single line engine
            if isempty(idx)
                error(['variable "' local_variable_names{i} '" was not found in the workspace but specified as a variable to load manually!'])
            else
                % add the variable to the selection
                selection = [selection idx];
            end
        end % end for
        ok = 1;
    end

    if ok == 1
        if ~is_interactive
            % Perform the non-interactive assignment
            for i = 1:length(selection)
                var = evalin('caller', vars_name{selection(i)});
                assignin('base', remote_variable_names{i}, var);
            end
        else
            % Interactive mode
            choice = questdlg('Would you like to store the variable(s) as their original name?', ...
	        'Save as', ...
	        'Yes','No','Yes');
            switch choice
                case 'Yes'
                    for n = selection
                        var = evalin('caller',vars_name{n});
                        assignin('base',vars_name{n},var);
                    end
                case 'No'
                    answer = inputdlg(vars_name(selection), 'New name', 1,vars_name(selection));
                    if ~isempty(answer)
                        m = 1;
                        for n = selection
                            var = evalin('caller',vars_name{n});
                            assignin('base',answer{m},var);
                            m = m + 1;
                        end
                    end
            end
        end
    end
end

