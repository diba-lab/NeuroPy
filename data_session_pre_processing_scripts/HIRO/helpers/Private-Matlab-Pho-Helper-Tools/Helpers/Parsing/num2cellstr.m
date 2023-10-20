function [cellStr] = num2cellstr(numericArray)
% num2cellstr: converts a numeric array to a cellstr representation
	cellStr = cellfun(@num2str, num2cell(numericArray), 'UniformOutput', false);
end