function outputParamStruct = fnSetStructureFields(targetFieldNames, desiredValues, userProvidedStruct, shouldOverwriteUserValues)
	% fnSetStructureFields: Adds the desiredValues{i} to the userProvidedStruct.(targetFieldNames{i}). Whether extant values are overwritten depends on shouldOverwriteUserValues
	%   Example usage:
	%       formatting = struct()
	%       formatting.color = 'black';
	%       formatting.fontweight = 'bold';
	%       formatting.fontsize = 24;
	%       formatting = fnSetStructureFields({'noses'}, {'true'}, formatting);
	%       formating when returned contains all the original field plus the field 'noses'

	% INPUTS: 
	% targetFieldNames: cell array of strings specifying the fields to be set
	% desiredValues: cell array of values with same length as requiredFieldNames
	% userProvidedStruct: an optional struct that will have the fields added or modified on it
	% shouldOverwriteUserValues: if true, the user-provided values will be overwritten with their respective desiredValues.

	num_fields = length(targetFieldNames);
	if (num_fields < length(desiredValues))
		error('not enough default values provided!')
	end

	if ~exist('userProvidedStruct', 'var')
		userProvidedStruct = struct(); % if no user provided struct is specified, make a blank one.
		shouldOverwriteUserValues = true; % If there was no user-provided struct, there's nothing to overwrite, so it's fine to set this to true.
	end

	outputParamStruct = userProvidedStruct;
	for i = 1:num_fields

		if shouldOverwriteUserValues
			% Sets the field of the user provided struct, overwriting its values if necessary.
			outputParamStruct.(targetFieldNames{i}) = desiredValues{i};
		else
			% If the field doesn't exist in the user provided struct, add it with the desired values.
			if ~isfield(outputParamStruct, targetFieldNames{i}) 
				outputParamStruct.(targetFieldNames{i}) = desiredValues{i};
			end
		end

	end % end for

end % fnSetStructureFields
