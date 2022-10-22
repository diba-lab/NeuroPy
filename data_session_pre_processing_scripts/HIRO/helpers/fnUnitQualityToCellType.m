function [speculated_unit_type, speculated_unit_contamination_level, speculated_unit_info] = fnUnitQualityToCellType(unitQuality)
%FNUNITQUALITYTOCELLTYPE Gets the unit qualities and turns them into the speculated cell types ({'pyramidal','contaminated','interneurons'})
% All units with qualities from 1-4 are pyramidal.
% The higher the number in this range, the higher is the contamination, so 1 and 2 are well-separated pyramidal units and if your analysis is not much sensitive to contaminations you can consider 3 and 4 as well. For my analysis, I considered 1 to 3. 8 and 9 are interneurons.
% 
% unitQuality is extracted from (for example) active_processing.spikes.quality
	speculated_unit_info = SpeculatedUnitInfo;
	[speculated_unit_type, speculated_unit_contamination_level] = SpeculatedUnitInfo.unitQualityToCellType(unitQuality);


	% speculated_unit_info.classNames = {'pyramidal','contaminated','interneurons'};
	% speculated_unit_info.classCutoffValues = [0 4 7 9];

	% num_units = length(active_processing.spikes.quality);

	% speculated_unit_type = discretize(active_processing.spikes.quality, ...
	% 	speculated_unit_info.classCutoffValues, ...
	% 	'categorical', speculated_unit_info.classNames);


	% speculated_unit_contamination_level = zeros([num_units 1]);
	% for i = 1:num_units
    %     temp.curr_type_index = grp2idx(speculated_unit_type(i));
    %     temp.curr_type_string = speculated_unit_info.classNames{temp.curr_type_index};
	% 	if (strcmpi(temp.curr_type_string, 'pyramidal'))
	% 		speculated_unit_contamination_level(i) = (active_processing.spikes.quality(i) - 1);
	% 		% value will be either [0, 1, 2, 3]

	% 	elseif (strcmpi(temp.curr_type_string, 'contaminated'))
	% 		%%% speculated_unit_contamination_level(i) = (active_processing.spikes.quality(i) - 5); % A value of 5 corresponds to minimal contamination for a contaminated cell, and 7 is the maximum
	% 		%%% value will be either [0, 1, 2]

	% 		% Keep contamination level relative to pyramidal to allow easier filtering
	% 		speculated_unit_contamination_level(i) = (active_processing.spikes.quality(i) - 1); % A value of 5 corresponds to minimal contamination for a contaminated cell, and 7 is the maximum
	% 		% value will be either [4, 5, 6]

	% 	elseif (strcmpi(temp.curr_type_string, 'interneurons'))
	% 		speculated_unit_contamination_level(i) = (active_processing.spikes.quality(i) - 8); % A value of 8 corresponds to minimal contamination for an interneuron, and 9 is the maximum
	% 		% value will be either [0, 1]

	% 	else
	% 		error('invalid!');
	% 	end

	% end

end

