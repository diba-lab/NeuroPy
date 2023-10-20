function [figH] = makeFigureActive(figH)
	%makeFigureActive: makes figH the active figure without stealing focus.
	%   When using figure(figH) to make an existing figure active it steals focus and is slow.
		if ~isempty(figH)
			% Get figure number:
			figNumber = figH.Number;                
			set(groot,'CurrentFigure',figNumber); % Make it active.
	%         set(0,'DefaultFigureVisible','off');
		end
end