function TF = isVariable(tbl, variableName)
	% isVariable: returns true if a table 'tbl' contains a named variable with the name 'variableName'
	TF = ismember(variableName, tbl.Properties.VariableNames);
end