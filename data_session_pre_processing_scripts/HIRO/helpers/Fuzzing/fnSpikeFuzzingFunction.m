function [blurredSpikeOutputs] = fnSpikeFuzzingFunction(samplingTimesteps, spikes, unitStatistics)
% convolve the function with each of the spikes:
    numUnits = length(spikes);

    config.timesteps = [0:0.1:20]; % 1x201
    config.decayConstants = (1 ./ unitStatistics.meanSpikeITI); %86x1
%     config.amplitudeConstants = ones([numUnits samplingTimesteps]); % 86 x samplingTimesteps (not used) % Currently a huge waste of memory
    
    % Convert back from seconds:
    samplingTimesteps = seconds(samplingTimesteps);
    
    config.endSamplingTimestamp = max(samplingTimesteps);
    
    
    % Want the function, on average, to decay to baseline before the average spike time occurs. This means it only accumulates when spikes are closer together than average for this unit
%     config.sourceFunction = @(unitIndex) config.amplitudeConstants(unitIndex) .* exp(config.decayConstants(unitIndex) .* config.timesteps);
    config.sourceFunction = exp(config.decayConstants * config.timesteps); % 86x201
    
    for i = 1:numUnits
%         blurredSpikeOutputs{i} = conv(spikes{i}, config.sourceFunction(i,:), 'same');
%         config.numSpikes(i) = length(spikes{i});
        blurredSpikeOutputs{i} = PiecewiseDrivingFunction(samplingTimesteps, spikes{i}, ones(size(spikes{i})));
    end
end