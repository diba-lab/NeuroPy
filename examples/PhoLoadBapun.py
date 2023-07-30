#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## Try adding NeuroPy repo path to the PythonPath
import sys
from pathlib import Path
print('sys.path: {}'.format(sys.path))
try:
    from neuropy import core
except ImportError:
    sys.path.append(r'C:\Users\Pho\repos\NeuroPy')
	# sys.path.append('/Users/pho/repo/Python Projects/NeuroPy')
    print('neuropy module not found, adding directory to sys.path. \nUpdated sys.path: {}'.format(sys.path))
	from neuropy import core


from neuropy.core.session.data_session_loader import DataSessionLoader

## Bapun Format:
# basedir = '/media/share/data/Bapun/Day5TwoNovel' # Linux
basedir = Path('R:\data\Bapun\Day5TwoNovel') # Windows
# basedir = '/Volumes/iNeo/Data/Bapun/Day5TwoNovel' # MacOS

sess = DataSessionLoader.bapun_data_session(basedir)
active_sess_config = sess.config
session_name = sess.name
print('session dataframe spikes: {}\n'.format(sess.spikes_df.shape))

print('session dataframe spikes: {}\nsession.neurons.n_spikes summed: {}\n'.format(sess.spikes_df.shape, np.sum(sess.neurons.n_spikes)))

# sess = sessions[0]

