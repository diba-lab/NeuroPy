import numpy as np
import pandas as pd

## Separate Runs on the Track

# Change of direction/inflection point by looking at the acceleration curve.

# velocity = np.insert(np.diff(pos), 0, 0)
# acceleration = np.insert(np.diff(velocity), 0, 0)


# Emphasize/hightlight poisition points within a specified time range


# Define Run:
	# Find all times the animal crosses the midline (the line bisecting the track through its midpoint) of the track.

def compute_lap_estimation(pos_df):
    # estimates the laps from the positions
	# pos_df at least has the columns 't', 'x'
	velocity = np.insert(np.diff(pos_df['x']), 0, 0)
	acceleration = np.insert(np.diff(velocity), 0, 0)

 
 

# Load from the 'traj' variable of an exported SpikeII.mat file:


## Direction-dependent tuning curves (find the direction of the animal at the time of each spike, bin them into 8 radial directions, and show the curves separately.


# I wonder if it follows a predictable cycle.

def get_lap_position(curr_lap_id):
    curr_position_df = sess.position.to_dataframe()
    curr_lap_t_start, curr_lap_t_stop = get_lap_times(curr_lap_id)
    print('lap[{}]: ({}, {}): '.format(curr_lap_id, curr_lap_t_start, curr_lap_t_stop))

    curr_lap_position_df_is_included = curr_position_df['t'].between(curr_lap_t_start, curr_lap_t_stop, inclusive=True) # returns a boolean array indicating inclusion in teh current lap
    curr_lap_position_df = curr_position_df[curr_lap_position_df_is_included] 
    # curr_position_df.query('-0.5 <= t < 0.5')
    curr_lap_position_traces = curr_lap_position_df[['x','y']].to_numpy().T
    print('\t {} positions.'.format(np.shape(curr_lap_position_traces)))
    # print('\t {} spikes.'.format(curr_lap_num_spikes))
    return curr_lap_position_traces



# Main Track Barrier Parts:
[63.5, 138.6] # bottom-left edge of the left-most track/platform barrier
[63.5, 144.2] # top-left edge of the left-most track/platform barrier
[223.9, 137.4] # bottom-right edge of the right-most track/platform barrier
[223.9, 150.0] # top-right edge of the right-most track/platform barrier


## Laps:
