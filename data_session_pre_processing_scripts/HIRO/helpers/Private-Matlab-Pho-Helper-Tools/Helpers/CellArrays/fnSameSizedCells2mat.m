function [stackedMatrixResult] = fnSameSizedCells2mat(cellArrayData)
%fnSameSizedCells2mat Converts a cell array containing same-sized vectors (either row or column) into a horizontally stacked matrix
%   Detailed explanation goes here

%% History:
% refactored from fnUnitDataCells2mat from Diba project on 2021-10-30

% cellArrayData: A 1xN or Nx1 cell array with each cell containing equal sized (Lx1 or 1xL) vectors

%% Returns:
% stackedMatrixResult: a vertically-stacked LxN matrix

	%% Extract all timeseries in the appropriate matrix format:
	% 1 x 126 cell should not be transposed.
	% 126 x 1 cell should
	if (size(cellArrayData, 1) > size(cellArrayData, 2))
		stackedMatrixResult = cell2mat(cellArrayData'); % 35351x126 double
	else
		stackedMatrixResult = cell2mat(cellArrayData); % 35351x126 double
	end


end
