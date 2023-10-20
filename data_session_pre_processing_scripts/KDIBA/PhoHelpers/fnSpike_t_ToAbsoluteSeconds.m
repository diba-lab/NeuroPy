function [Spktimes_seconds, Spktimes_microseconds] = fnSpike_t_ToAbsoluteSeconds(spkt, tbegin_uSec, SampleRateHz)
    %fnSpike_t_ToAbsoluteSeconds spike.t to absolute seconds:
    %   Detailed explanation goes here
    Spktimes_microseconds = tbegin_uSec + ((spkt * 1e6)/SampleRateHz); % known to be in units of microseconds
    Spktimes_seconds = Spktimes_microseconds / 1e6; % This converts the spike.t column into absoulte seconds
end