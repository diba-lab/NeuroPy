function [cellArray] = fnCellContentsTranpose(cellArray)
	% fnCellContentsTranpose: takes the transpose of the contents of each cell in the cellArray. Does not change the shape of cellArray itself
	cellArray = cellfun(@transpose, cellArray, 'UniformOutput', false);
end