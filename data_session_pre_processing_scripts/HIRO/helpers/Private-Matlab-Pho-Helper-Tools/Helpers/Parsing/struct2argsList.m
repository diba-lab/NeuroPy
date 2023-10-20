function paramCell = struct2argsList(paramStruct)
	% struct2argsList: Converts structure to parameter-value pairs
	%   Example usage:
	%       formatting = struct()
	%       formatting.color = 'black';
	%       formatting.fontweight = 'bold';
	%       formatting.fontsize = 24;
	%       formatting = struct2argsList(formatting);
	%       xlabel('Distance', formatting{:});
	% Adapted from:
	% http://stackoverflow.com/questions/15013026/how-can-i-unpack-a-matlab-structure-into-function-arguments
	% by user 'yuk'

	fname = fieldnames(paramStruct);
	fval = struct2cell(paramStruct);
	paramCell = [fname, fval]';
	paramCell = paramCell(:);
end % struct2argsList
