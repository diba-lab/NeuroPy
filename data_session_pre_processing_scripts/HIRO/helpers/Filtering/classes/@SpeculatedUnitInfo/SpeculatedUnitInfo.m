classdef SpeculatedUnitInfo
    %SPECULATED_UNIT_INFO defines cell type ({'pyramidal', 'contaminated', 'interneurons'}) colors: 
    %   Detailed explanation goes here
%     enumeration
%           pyramidal
%           contaminated
%           interneurons
%     end

    properties (Constant)
      classColors = [0.8, 0.5, 0.1; 0.5, 0.1, 0.1; 0.0, 0.7, 0.7];
      classNames = {'pyramidal','contaminated','interneurons'};
      classCutoffValues = [0 4 7 9];
    end % end Constant properties block

    methods
        function obj = SpeculatedUnitInfo()
            %SPECULATED_UNIT_INFO Construct an instance of this class
            %   Detailed explanation goes here
        end
    end

    methods (Static)
        [speculated_unit_type, speculated_unit_contamination_level, SpeculatedUnitInfo] = unitQualityToCellType(unit_quality) % function signature required to match the one in the uniQualityToCellType.m file 

%       function out = setgetVar(data)
%          persistent Var;
%          if nargin
%             Var = data;
%          end
%          out = Var;
%       end

    end % end static method block


end

