function [outputFilterFunction, functionCells] = fnBuildPeriodDetectingFilter(startTimes, endTimes)
%FNBUILDPERIODDETECTINGFILTER Should return a function that takes a single event time as input and returns the period index it falls within if it does fall within one and nan otherwise
%   Detailed explanation goes here

    %% Function Start:
    num_periods = length(startTimes);
    functionCells = cell(num_periods,1);
    
    for i = 1:num_periods
        currentEventFallsWithinPeriodFunction = @(t) ((startTimes(i) <= t) & (t <= endTimes(i)));
        functionCells{i} = currentEventFallsWithinPeriodFunction;
    end

    % outputFilterFunction
    outputFilterFunction = @(times) cellfun(@(period_comparison_fcn) find(period_comparison_fcn(times)~=0, 1, 'first'), functionCells, 'UniformOutput', false);

    % times = spikesTable.time{unit_idx}

    % unit_matches = cellfun(@(period_comparison_fcn) find(period_comparison_fcn(spikesTable.time{unit_idx})~=0, 1, 'first'), functionCells, 'UniformOutput', false);
    
end

