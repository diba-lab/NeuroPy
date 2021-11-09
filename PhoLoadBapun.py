#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

## Try adding NeuroPy repo path to the PythonPath
import sys
from pathlib import Path
print('sys.path: {}'.format(sys.path))
try:
    from neuropy import core
except ImportError:
    # sys.path.append(r'C:\Users\Pho\repos\NeuroPy')
	sys.path.append('/Users/pho/repo/Python Projects/NeuroPy')
    print('neuropy module not found, adding directory to sys.path. \nUpdated sys.path: {}'.format(sys.path))
	from neuropy import core


# from callfunc import processData
from ProcessData import ProcessData

#%% Subjects
basePath = [
	"/Volumes/iNeo/Data/Bapun/Day5TwoNovel/"
]
# basePath = [
#     "/data/Clustering/SleepDeprivation/RatN/Day2/",
#     "/data/Clustering/SleepDeprivation/RatK/Day4/"
#     "/data/Clustering/SleepDeprivation/RatN/Day4/"
# ]



sessions = [ProcessData(_) for _ in basePath
# sess = sessions[0]

