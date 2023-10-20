function [startTime, endTime] = fnJumpToLap(lapsMatrix, lapIndex)
    % fnJumpToLap: tries to jump the current plot to the time range of a given lap
   % lapsMatrix: 232x3 double array
   %%   as can be loaded by:
        %    if ~exist('lapsInfo','var')
        %     lapsInfo = load('/Volumes/iNeo/Data/Rotation_3_Kamran Diba Lab/DataProcessingProject/Hiro_Datasets/analysesResults_13-Oct-2021/Roy-maze1/TrackLaps/trackLaps.mat', 'laps', 'lapsStruct');
        %     %% laps are represented in absolute timestamps, convert to experiment relative timestamps
        %         % outputs will be aligned with the timestamps in the active_processing.position_table's timestamp column
        %     lapsInfo.laps(:, 1:2) = ((lapsInfo.laps(:, 1:2) - fileinfo.tbegin) ./ 1e6); % Convert to relative timestamps since start
        %     lapsInfo.numTotalLaps = size(lapsInfo.laps, 1); 
        %    end
        
       
   curr_cellLapIndex = 15; xlim([obj.Loaded.lapsTable.lapStartTime(curr_cellLapIndex), obj.Loaded.lapsTable.lapEndTime(curr_cellLapIndex)]);

   startTime = lapsMatrix(lapIndex, 1);
   endTime = lapsMatrix(lapIndex, 2);
   xlim([startTime endTime])
end