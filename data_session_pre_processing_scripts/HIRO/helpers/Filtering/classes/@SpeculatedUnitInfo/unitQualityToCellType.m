function [speculated_unit_type, speculated_unit_contamination_level] = unitQualityToCellType(unit_quality)
    % unitQualityToCellType: converted from standalone function 'fnUnitQualityToCellType.m' from Diba project
    % unit_quality: unit_quality
%     import SpeculatedUnitInfo.*;

    classCutoffValues = [0 4 7 9]; % SpeculatedUnitInfo.classCutoffValues
    classNames = {'pyramidal','contaminated','interneurons'}; % SpeculatedUnitInfo.classNames

    num_units = length(unit_quality);

    nan_quality_units = find(isnan(unit_quality));
    if ~isempty(nan_quality_units)
        unit_quality(nan_quality_units) = 7; % setting NaN quality units to maximally contaminated
        warning(sprintf('Have %f units with NaN quality! Setting quality to maxiimally contaminated: %f', length(nan_quality_units), nan_quality_units));
    end

    speculated_unit_type = discretize(unit_quality, ...
        classCutoffValues, ...
        'categorical', classNames);

    speculated_unit_contamination_level = zeros([num_units 1]);
    for i = 1:num_units
        temp.curr_type_index = grp2idx(speculated_unit_type(i));
        temp.curr_type_string = classNames{temp.curr_type_index};
        if (strcmpi(temp.curr_type_string, 'pyramidal'))
	        speculated_unit_contamination_level(i) = (unit_quality(i) - 1);
	        % value will be either [0, 1, 2, 3] 
        elseif (strcmpi(temp.curr_type_string, 'contaminated'))
	        %%% speculated_unit_contamination_level(i) = (unit_quality(i) - 5); % A value of 5 corresponds to minimal contamination for a contaminated cell, and 7 is the maximum
	        %%% value will be either [0, 1, 2]
	        % Keep contamination level relative to pyramidal to allow easier filtering
	        speculated_unit_contamination_level(i) = (unit_quality(i) - 1); % A value of 5 corresponds to minimal contamination for a contaminated cell, and 7 is the maximum
	        % value will be either [4, 5, 6]
        elseif (strcmpi(temp.curr_type_string, 'interneurons'))
	        speculated_unit_contamination_level(i) = (unit_quality(i) - 8); % A value of 8 corresponds to minimal contamination for an interneuron, and 9 is the maximum
	        % value will be either [0, 1]   
        else
	        error('invalid!');
        end        
        
    end
end