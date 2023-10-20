function [row_index, col_index] = ind2subplot(numRows, numColumns, curr_linear_subplot_index)
%ind2subplot Multiple subplot subscripts from linear index.
%   ind2subplot is used to determine the equivalent subplot subscript values
%   corresponding to a given single index into an array. Note that these differ from the normal matlab index notations. 
%   Copyright 2020 Pho Hale
	vi = rem(curr_linear_subplot_index - 1, numColumns) + 1;
	row_index = double((curr_linear_subplot_index - vi)/numColumns + 1);
	col_index = double(vi);
end
