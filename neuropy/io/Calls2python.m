function Calls2python(detection_file)
%Calls2python Saves DeepSqueak detection file in a format that can be
%imported to python in NeuroPy. Saves in same directory as input file.
% Read into python using the neuropy.io.usvio DeepSqueakIO class.
%   
%   Calls2python(detection_file_path)

% Convert file to cells that can be read by python
calls_data = table2cell(load(detection_file).Calls);
calls_keys = load(detection_file).Calls.Properties.VariableNames;

[path, name, ~] = fileparts(detection_file);

save(fullfile(path, strcat(name, '_cell.mat')), 'calls_keys', 'calls_data')
end