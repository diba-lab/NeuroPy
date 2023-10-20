function [mergedStruct] = fnMergeStructs(varargin)
	% fnMergeStructs: Combines structures with unlike fields into a single merged structure
	%   Example usage:

	if nargin == 0
		mergedStruct = struct;
	elseif nargin == 1
		mergedStruct = varargin{1}; % it better be a struct tho
	else
		mergedStruct = struct; % start with an empty struct
		for additional_struct_idx = 1:length(varargin)
			currStruct = varargin{additional_struct_idx};
			% currStruct2argsList = varargin{additional_struct_idx}
			fnames = fieldnames(currStruct);
			% fvals = struct2cell(currStruct);
			for aNameIdx = 1:length(fnames)
                aName = fnames{aNameIdx};
				mergedStruct.(aName) = currStruct.(aName);
			end % end for aName = fnames

		end % end for additional structs
	end % end if

end % fnMergeStructs
