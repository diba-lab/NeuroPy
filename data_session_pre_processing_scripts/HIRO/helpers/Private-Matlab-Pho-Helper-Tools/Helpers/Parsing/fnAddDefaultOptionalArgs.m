function outputParamStruct = fnAddDefaultOptionalArgs(requiredFieldNames, defaultValues, userProvidedParamStruct)
	% fnAddDefaultOptionalArgs: Adds provided default values if the user didn't set them, otherwise it prefers the user's version

%% Example:
	% if ~exist('labelOptions','var')
	% 	labelOptions = struct; 
	% end
	% labelOptions = fnAddDefaultOptionalArgs({'shouldIncludeFullTemperatureSpecString'}, ...
	% 			{false}, ...
	% 			labelOptions);

	% INPUTS: 
	% requiredFieldNames: cell array of strings specifying the parameters that must be added
	% defaultValues: cell array of default values with same length as requiredFieldNames
	% userProvidedParamStruct: an optional struct


	% call fnSetStructureFields with the option to overwrite user provided values set to false
	outputParamStruct = fnSetStructureFields(requiredFieldNames, defaultValues, userProvidedParamStruct, false); 

end % fnAddDefaultOptionalArgs
