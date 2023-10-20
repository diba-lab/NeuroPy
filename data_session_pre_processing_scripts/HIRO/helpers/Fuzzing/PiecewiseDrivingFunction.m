function z = PiecewiseDrivingFunction(t, spikeOffsetsInSeconds, spikeAmplitudeValues)
    num_spikes = length(spikeAmplitudeValues);
    functionCells = cell(num_spikes,1);
    
    for i=1:num_spikes
        currTimeOffset = spikeOffsetsInSeconds(i);
        
        currentSpikeImpulseFunction = @(t) (spikeAmplitudeValues(i) .* ((t-currTimeOffset)>=0) .* exp(-(t - currTimeOffset)));
        functionCells{i} = currentSpikeImpulseFunction;
    end
  
    % Iterate over all functions and compute t
    z = zeros(size(t));
    for iFcn = 1:num_spikes
        temp_spikeContribution = functionCells{iFcn}(t);
        temp_good_spike_contribution_indicies = ~isnan(temp_spikeContribution);
        z(1,temp_good_spike_contribution_indicies) = z(1,temp_good_spike_contribution_indicies) + temp_spikeContribution(temp_good_spike_contribution_indicies);
    end
      
end

