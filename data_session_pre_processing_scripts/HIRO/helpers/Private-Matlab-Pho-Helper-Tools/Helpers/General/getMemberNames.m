function [propNames] = getMemberNames(inputObj)
%getMemberNames Gets the names of the member variables/ fields of inputObj
%% Inputs:
% inputObj: either a class or struct

%% Outputs:
% propNames: a cell array of property names
    if isstruct(inputObj)
        % It's trivial if a struct:
        propNames = fieldnames(inputObj);
    elseif isobject(inputObj)
        allprops = properties(inputObj);
        propNames = cell([numel(allprops) 1]);
        for i=1:numel(allprops)
            m = findprop(inputObj, allprops{i});
            propNames{i} = m.Name;
        end
    else
        error('inputObj must be either a struct or object');
    end % end if
            
end % end getMemberVariableNames

