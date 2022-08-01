function [figH, wasNewFigCreated] = createFigureWithNameIfNeeded(name)
% createFigureWithNameIfNeeded: If an existing open figure with the provided name exists, it finds it, makes it active, and then returns it. Otherwise it makes a new figure and returns that.
    figHPotential = findobj('Type', 'Figure', 'Name', name);
    if isempty(figHPotential)
        % Make a new figure
        wasNewFigCreated = true;
       figH = figure('Name', name, 'NumberTitle','off'); % Make a new figure
    else
        wasNewFigCreated = false;
        % Use existing figure
        figH = figHPotential;
        % figure(figH); % Make it active.
		figH = makeFigureActive(figH); % Make it active without stealing focus
    end

end